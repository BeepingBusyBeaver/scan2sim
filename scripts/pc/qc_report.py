# scripts/pc/qc_report.py
import argparse
import numpy as np
from pathlib import Path
import open3d as o3d

from scipy.spatial import cKDTree
from scripts.common.io_paths import (
    DATA_REAL_HUMAN_EXAMPLE,
    DATA_VIRTUAL_POSE324_EXAMPLE,
    resolve_repo_path,
)


# -----------------------
# File readers
# -----------------------
def read_ply_xyz(path: str) -> np.ndarray:
    """
    ASCII PLY (x y z only or with more properties) from Unity exporter.
    Returns Nx3 float64 array.
    """
    path = Path(path)
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading PLY header")
            s = line.decode("utf-8", errors="replace").rstrip("\n")
            header.append(s)
            if s.strip() == "end_header":
                break

        n_verts = None
        prop_names = []
        in_vertex = False
        for s in header:
            if s.startswith("element vertex"):
                n_verts = int(s.split()[-1])
                in_vertex = True
                continue
            if s.startswith("element") and not s.startswith("element vertex"):
                in_vertex = False
            if in_vertex and s.startswith("property"):
                prop_names.append(s.split()[-1])

        if n_verts is None:
            raise ValueError("PLY header missing 'element vertex'")

        ix, iy, iz = prop_names.index("x"), prop_names.index("y"), prop_names.index("z")

        pts = []
        for _ in range(n_verts):
            line = f.readline()
            if not line:
                break
            parts = line.decode("utf-8", errors="replace").strip().split()
            pts.append([float(parts[ix]), float(parts[iy]), float(parts[iz])])

    return np.asarray(pts, dtype=np.float64)


def read_pcd(path: str):
    """
    PCD v0.7 ASCII or binary (uncompressed) basic reader.
    Returns:
      header_lines, header_dict, expanded_fields, structured_array
    """
    path = Path(path)
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("EOF before DATA in PCD")
            s = line.decode("utf-8", errors="replace").strip()
            header_lines.append(s)
            if s.startswith("DATA"):
                data_spec = s.split()[1].lower()
                break

        header = {}
        for s in header_lines:
            if not s or s.startswith("#"):
                continue
            k = s.split()[0]
            header[k] = s[len(k):].strip()

        fields = header["FIELDS"].split()
        size = list(map(int, header["SIZE"].split()))
        typ = header["TYPE"].split()
        count = list(map(int, header.get("COUNT", " ".join(["1"] * len(fields))).split()))
        points = int(header["POINTS"])

        type_map = {
            ("F", 4): np.float32, ("F", 8): np.float64,
            ("U", 1): np.uint8,  ("U", 2): np.uint16, ("U", 4): np.uint32,
            ("I", 1): np.int8,   ("I", 2): np.int16,  ("I", 4): np.int32,
        }

        dtype_fields = []
        expanded_fields = []
        for i, name in enumerate(fields):
            c = count[i]
            dt = type_map[(typ[i], size[i])]
            for j in range(c):
                n = name if c == 1 else f"{name}_{j}"
                expanded_fields.append(n)
                dtype_fields.append((n, dt))
        dtype = np.dtype(dtype_fields)

        if data_spec == "ascii":
            lines = f.read().decode("utf-8", errors="replace").strip().splitlines()
            arr = np.zeros((len(lines), len(expanded_fields)), dtype=np.float64)
            for i, ln in enumerate(lines):
                parts = ln.strip().split()
                arr[i, :len(parts)] = [float(x) for x in parts[:len(expanded_fields)]]

            out = np.zeros(len(lines), dtype=dtype)
            for j, n in enumerate(expanded_fields):
                out[n] = arr[:, j].astype(dtype[n])
            return header_lines, header, expanded_fields, out

        elif data_spec == "binary":
            raw = f.read()

            point_step = sum(size[i] * count[i] for i in range(len(fields)))
            expected = points * point_step
            if len(raw) < expected:
                points = len(raw) // point_step
                expected = points * point_step
            raw = raw[:expected]

            # packed dtype with explicit offsets if needed
            offsets = []
            off = 0
            names = []
            formats = []
            for i, name in enumerate(fields):
                dt = type_map[(typ[i], size[i])]
                for j in range(count[i]):
                    n = name if count[i] == 1 else f"{name}_{j}"
                    names.append(n)
                    formats.append(dt)
                    offsets.append(off)
                    off += np.dtype(dt).itemsize

            packed_dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": point_step})
            out = np.frombuffer(raw, dtype=packed_dtype, count=points)
            return header_lines, header, expanded_fields, out

        else:
            raise ValueError(f"Unsupported PCD DATA format: {data_spec}")

def read_any_xyz(path: str) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(path))  # .ply/.pcd 자동 인식
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.size == 0:
        raise ValueError(f"Empty point cloud: {path}")
    return pts

# -----------------------
# Metrics
# -----------------------
def knn_distance_stats(pts: np.ndarray, k: int = 8):
    """
    kNN 거리의 분포를 봅니다.
    - kth 거리의 '절대값'은 스케일(단위)에 좌우됨
    - p95/median, p99/median 같은 비율은 스케일 영향이 적어 비교에 유리
    """
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=k+1, workers=-1)  # 0번째는 자기자신(거리 0)
    kth = dists[:, k]
    med = np.median(kth)
    p95 = np.percentile(kth, 95)
    p99 = np.percentile(kth, 99)
    return {
        "k": k,
        "median": float(med),
        "p95": float(p95),
        "p99": float(p99),
        "p95_over_median": float(p95 / med) if med > 0 else np.nan,
        "p99_over_median": float(p99 / med) if med > 0 else np.nan,
    }


def outlier_fraction_kdist(pts: np.ndarray, k: int = 8, thresh_mult: float = 3.0):
    """
    단순 outlier proxy:
    kth-neighbor 거리 > (thresh_mult * median) 인 점의 비율
    """
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=k+1, workers=-1)
    kth = dists[:, k]
    med = np.median(kth)
    frac = float(np.mean(kth > thresh_mult * med))
    return {"k": k, "thresh_mult": thresh_mult, "median": float(med), "fraction": frac}


def local_pca_features(pts: np.ndarray, k: int = 20):
    """
    각 점의 k-neighborhood 공분산 고유값(λ1≥λ2≥λ3)로 로컬 형상을 요약.
      - surface_variation = λ3 / (λ1+λ2+λ3) : 표면 두께/잡음 대리 (작을수록 얇은 표면)
      - planarity = (λ2-λ3) / λ1           : 평면성 (클수록 로컬이 평면에 가까움)
      - scattering = λ3 / λ1               : 가장 얇은 축 두께 비율 (클수록 3D로 퍼짐)
    """
    tree = cKDTree(pts)
    _, idxs = tree.query(pts, k=k, workers=-1)

    n = pts.shape[0]
    surface_var = np.empty(n, dtype=np.float64)
    planarity = np.empty(n, dtype=np.float64)
    scattering = np.empty(n, dtype=np.float64)

    for i in range(n):
        neigh = pts[idxs[i]]
        c = neigh.mean(axis=0)
        X = neigh - c
        cov = (X.T @ X) / max(len(neigh) - 1, 1)
        w = np.linalg.eigvalsh(cov)  # ascending
        l1, l2, l3 = w[2], w[1], w[0]
        s = l1 + l2 + l3

        surface_var[i] = (l3 / s) if s > 0 else 0.0
        planarity[i] = ((l2 - l3) / l1) if l1 > 0 else 0.0
        scattering[i] = (l3 / l1) if l1 > 0 else 0.0

    def summarize(a):
        return {"median": float(np.median(a)), "mean": float(np.mean(a)), "p95": float(np.percentile(a, 95))}

    return {
        "k": k,
        "surface_variation": summarize(surface_var),
        "planarity": summarize(planarity),
        "scattering": summarize(scattering),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quality-control report for virtual/reality point clouds.")
    parser.add_argument("--virtual", type=str, default=DATA_VIRTUAL_POSE324_EXAMPLE, help="Virtual point cloud path")
    parser.add_argument("--reality", type=str, default=DATA_REAL_HUMAN_EXAMPLE, help="Reality point cloud path")
    parser.add_argument("--knn-k", type=int, default=8, help="k for kNN distance stats")
    parser.add_argument("--pca-k", type=int, default=20, help="k for local PCA features")
    parser.add_argument("--outlier-mult", type=float, default=3.0, help="Multiplier for k-distance outlier fraction")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[2]

    virtual_path = resolve_repo_path(project_root, args.virtual)
    reality_path = resolve_repo_path(project_root, args.reality)

    virtual_pts = read_any_xyz(virtual_path)
    reality_pts = read_any_xyz(reality_path)

    print("=== VIRTUAL ===")
    print("points:", virtual_pts.shape[0])
    print("knn:", knn_distance_stats(virtual_pts, k=args.knn_k))
    print("outlier_fraction:", outlier_fraction_kdist(virtual_pts, k=args.knn_k, thresh_mult=args.outlier_mult))
    print("local_pca:", local_pca_features(virtual_pts, k=args.pca_k))

    print("\n=== REALITY ===")
    print("points:", reality_pts.shape[0])
    print("knn:", knn_distance_stats(reality_pts, k=args.knn_k))
    print("outlier_fraction:", outlier_fraction_kdist(reality_pts, k=args.knn_k, thresh_mult=args.outlier_mult))
    print("local_pca:", local_pca_features(reality_pts, k=args.pca_k))


if __name__ == "__main__":
    main()

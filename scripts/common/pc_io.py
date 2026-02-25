# scripts/common/pc_io.py
from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable

import numpy as np
import open3d as o3d

# --- PCD(mm, back/down/right) -> PLY(m, left/back/up) ---
S_MM_TO_M = 1e-3
R_PCD_TO_PLY = np.array(
    [
        [0.0, 0.0, -1.0],  # x_ply = -z_pcd
        [1.0, 0.0,  0.0],  # y_ply =  x_pcd
        [0.0, -1.0, 0.0],  # z_ply = -y_pcd
    ],
    dtype=np.float64,
)
T_PCD_TO_PLY = np.eye(4, dtype=np.float64)
T_PCD_TO_PLY[:3, :3] = S_MM_TO_M * R_PCD_TO_PLY


def _fallback_read_pcd_ascii_xyz_rgb(path: Path) -> o3d.geometry.PointCloud:
    """Minimal ASCII .pcd reader (xyz + optional rgb uint32)."""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    data_idx = None
    fields = None
    for i, ln in enumerate(lines):
        if ln.startswith("FIELDS"):
            fields = ln.split()[1:]
        if ln.strip().startswith("DATA"):
            if "ascii" not in ln:
                raise ValueError(f"Fallback reader supports only ASCII PCD. Got: {ln}")
            data_idx = i + 1
            break
    if data_idx is None:
        raise ValueError("PCD header missing 'DATA ascii' line.")

    arr = np.loadtxt(lines[data_idx:], dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]

    if fields is None:
        xyz = arr[:, :3]
        rgb = arr[:, 3] if arr.shape[1] > 3 else None
    else:
        idx = {f: j for j, f in enumerate(fields)}
        xyz = arr[:, [idx["x"], idx["y"], idx["z"]]]
        rgb = arr[:, idx["rgb"]] if "rgb" in idx else None

    good = np.isfinite(xyz).all(axis=1)
    xyz = xyz[good]
    rgb = rgb[good] if rgb is not None else None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if rgb is not None:
        rgb_u = rgb.astype(np.uint32)
        r = ((rgb_u >> 16) & 255).astype(np.float64)
        g = ((rgb_u >> 8) & 255).astype(np.float64)
        b = (rgb_u & 255).astype(np.float64)
        colors = np.stack([r, g, b], axis=1) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def read_point_cloud_any(path: Path) -> o3d.geometry.PointCloud:
    """Try Open3D loader first; fallback to minimal ASCII PCD parser."""
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) > 0:
        return pcd
    if path.suffix.lower() == ".pcd":
        return _fallback_read_pcd_ascii_xyz_rgb(path)
    raise ValueError(f"Failed to read point cloud: {path}")


def has_glob_chars(s: str) -> bool:
    return any(ch in s for ch in ["*", "?", "["])


def expand_inputs(root: Path, items: Iterable[str], exts: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    exts_set = {e.lower() for e in exts}

    for raw in items:
        raw_path = Path(raw)
        abs_like = raw if raw_path.is_absolute() else str((root / raw_path))

        matches: list[Path] = []
        if has_glob_chars(abs_like):
            for m in glob.glob(abs_like, recursive=True):
                matches.append(Path(m).resolve())
        else:
            p = Path(abs_like).resolve()
            if p.is_dir():
                for e in exts_set:
                    matches.extend(sorted(p.glob(f"*{e}")))
            else:
                matches.append(p)

        for p in matches:
            p = p.resolve()
            if not p.exists() or not p.is_file():
                continue
            if p.suffix.lower() not in exts_set:
                continue
            if p not in seen:
                seen.add(p)
                out.append(p)

    out.sort(key=lambda x: str(x))
    return out


def apply_pcd_transform_if_needed(
    pcd: o3d.geometry.PointCloud, path: Path, apply: bool = True
) -> o3d.geometry.PointCloud:
    if apply and path.suffix.lower() == ".pcd":
        pcd = o3d.geometry.PointCloud(pcd)
        pcd.transform(T_PCD_TO_PLY)
    return pcd

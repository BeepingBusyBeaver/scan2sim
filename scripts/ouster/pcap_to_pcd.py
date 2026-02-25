# scripts/ouster/pcap_to_pcd.py
from __future__ import annotations

import argparse
from pathlib import Path

from ouster.sdk import core, pcap

from scripts.common.io_paths import DATA_REAL_RAW_DIR, resolve_repo_path

"""
python -m scripts.ouster.pcap_to_pcd \
  data/labeling/001/simpleMotion_wonjin.pcap \
  --scan 861 \
  --outdir data/real/raw \
  --base SN001
"""


def pcap_to_pcd_select_scans(
    source_file: Path,
    scan_indices,
    pcd_dir: Path,
    pcd_base: str = "SN001",
    pcd_ext: str = "pcd",
) -> None:
    try:
        import open3d as o3d  # type: ignore
    except ModuleNotFoundError:
        raise RuntimeError("open3d 필요함. `pip install open3d` ㄱㄱ")

    scan_indices_sorted = sorted(set(int(index) for index in scan_indices))
    if not scan_indices_sorted:
        print("scan_indices 비어있음")
        return

    pcd_dir.mkdir(parents=True, exist_ok=True)

    source = pcap.PcapScanSource(str(source_file))
    metadata = source.sensor_info[0]

    if metadata.format.udp_profile_lidar == core.UDPProfileLidar.RNG19_RFL8_SIG16_NIR16_DUAL:
        print("Note: dual returns pcap임. 예제는 second return 무시함.")

    xyzlut = core.XYZLut(metadata)

    wanted = set(scan_indices_sorted)
    max_wanted = scan_indices_sorted[-1]

    for scan_idx, scan_list in enumerate(source):
        if scan_idx > max_wanted:
            break
        if scan_idx not in wanted:
            continue

        for sub_idx, scan in enumerate(scan_list):
            if scan is None:
                continue

            xyz = xyzlut(scan.field(core.ChanField.RANGE))

            point_cloud = o3d.geometry.PointCloud()  # type: ignore
            point_cloud.points = o3d.utility.Vector3dVector(xyz.reshape(-1, 3))  # type: ignore

            pcd_path = pcd_dir / f"{pcd_base}_{scan_idx:06d}.{pcd_ext}"
            print(f"write scan #{scan_idx} (sub {sub_idx}) -> {pcd_path}")
            o3d.io.write_point_cloud(str(pcd_path), point_cloud)  # type: ignore

    source.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pcap_path")
    parser.add_argument("--scan", type=int, nargs="+", required=True, help="원하는 scan index들 (예: --scan 0 10 20)")
    parser.add_argument("--outdir", default=DATA_REAL_RAW_DIR)
    parser.add_argument("--base", default="SN001")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    pcap_path = resolve_repo_path(repo_root, args.pcap_path)
    outdir = resolve_repo_path(repo_root, args.outdir)

    if not pcap_path.exists():
        raise ValueError(f"PCAP not found: {pcap_path}")

    pcap_to_pcd_select_scans(
        source_file=pcap_path,
        scan_indices=args.scan,
        pcd_dir=outdir,
        pcd_base=args.base,
        pcd_ext="pcd",
    )


if __name__ == "__main__":
    main()

# scripts/ouster/pcap_to_ply.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import open3d as o3d

from ouster.sdk import core, open_source
from ouster.sdk.util import metadata as meta_util
from scripts.common.io_paths import DATA_LABELING_PCAP_FRAMES_DIR

"""
Convert Ouster PCAP + metadata into per-scan PLY files.

Examples:
  python -m scripts.ouster.pcap_to_ply \
    --pcap data/labeling/001/simpleMotion_wonjin.pcap \
    --meta data/labeling/001/simpleMotion_wonjin.json \
    --out_dir data/labeling/001/pcap_frames/test1 \
    --start 0 --count 100 --stride 1

  # auto-resolve metadata if .json shares prefix with .pcap
  python -m scripts.ouster.pcap_to_ply \
    --pcap data/labeling/001/simpleMotion_wonjin.pcap \
    --out_dir data/labeling/001/pcap_frames/test1
"""


def _iter_scans(source, sensor_idx: int) -> Iterable[core.LidarScan]:
    for scans in source:
        if isinstance(scans, (list, tuple)):
            if sensor_idx < 0:
                scan = scans[0] if scans else None
            else:
                scan = scans[sensor_idx] if sensor_idx < len(scans) else None
        else:
            scan = scans
        if scan is None:
            continue
        yield scan


def _make_xyzlut(source, sensor_idx: int, use_extrinsics: bool) -> core.XYZLut:
    infos = getattr(source, "sensor_info", None)
    if infos is None or len(infos) == 0:
        raise ValueError("Failed to access sensor_info from source (metadata missing?).")
    idx = 0 if sensor_idx < 0 else sensor_idx
    if idx >= len(infos):
        raise ValueError(f"sensor_idx {idx} out of range (available: {len(infos)})")
    return core.XYZLut(infos[idx], use_extrinsics=use_extrinsics)


def _scan_to_points(scan: core.LidarScan, xyzlut: core.XYZLut) -> np.ndarray:
    rng = scan.field(core.ChanField.RANGE)
    xyz = xyzlut(rng)
    xyz = xyz.reshape((-1, 3))
    r = rng.reshape((-1,))
    mask = np.isfinite(xyz).all(axis=1) & (r > 0)
    return xyz[mask]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap", type=str, required=True, help="Input .pcap path")
    ap.add_argument("--meta", type=str, default=None, help="Optional metadata .json path")
    ap.add_argument("--out_dir", type=str, default=DATA_LABELING_PCAP_FRAMES_DIR, help="Output directory")
    ap.add_argument("--base", type=str, default=None, help="Output basename prefix")
    ap.add_argument("--start", type=int, default=0, help="Start scan index")
    ap.add_argument("--count", type=int, default=0, help="Number of scans to save (0=all)")
    ap.add_argument("--stride", type=int, default=1, help="Stride between scans")
    ap.add_argument("--sensor_idx", type=int, default=0, help="Sensor index (0 for single sensor)")
    ap.add_argument("--use_extrinsics", action="store_true", help="Apply extrinsics if present")
    ap.add_argument("--ascii", action="store_true", help="Write PLY in ASCII format")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    pcap_path = Path(args.pcap)
    if not pcap_path.is_absolute():
        pcap_path = root / pcap_path
    if not pcap_path.exists():
        raise ValueError(f"PCAP not found: {pcap_path}")

    meta_path: Optional[str] = None
    if args.meta:
        mp = Path(args.meta)
        if not mp.is_absolute():
            mp = root / mp
        meta_path = str(mp)
    else:
        meta_path = meta_util.resolve_metadata(str(pcap_path))

    meta_list = [meta_path] if meta_path else None
    source = open_source(str(pcap_path), meta=meta_list, sensor_idx=args.sensor_idx, collate=True)

    xyzlut = _make_xyzlut(source, args.sensor_idx, use_extrinsics=args.use_extrinsics)

    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base = args.base if args.base else pcap_path.stem

    saved = 0
    for i, scan in enumerate(_iter_scans(source, args.sensor_idx)):
        if i < args.start:
            continue
        if (i - args.start) % max(args.stride, 1) != 0:
            continue
        if args.count and saved >= args.count:
            break

        pts = _scan_to_points(scan, xyzlut)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        out_path = out_dir / f"{base}_{i:06d}.ply"
        ok = o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=args.ascii)
        if not ok:
            raise RuntimeError(f"Failed to write: {out_path}")
        saved += 1

        if saved % 10 == 0:
            print(f"[INFO] saved {saved} scans... (latest: {out_path.name})")

    print(f"[DONE] saved {saved} scans to {out_dir}")


if __name__ == "__main__":
    main()

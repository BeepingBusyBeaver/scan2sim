# scripts/pc/convert_pcd2ply.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

"""
python -m scripts.pc.convert_pcd2ply \
  --in_pcd data/real/raw/SN001_000861.pcd \
  --out data/interim/pcd2ply/SN001_000861.ply

python -m scripts.pc.convert_pcd2ply \
  --in_pcd "data/real/raw/SN001_*.pcd" \
  --out_dir data/interim/pcd2ply
"""

from scripts.common.pc_io import (
    T_PCD_TO_PLY,
    expand_inputs,
    read_point_cloud_any,
)
from scripts.common.io_paths import (
    DATA_INTERIM_PCD2PLY_DIR,
    DATA_REAL_RAW_EXAMPLE,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_pcd",
        type=str,
        nargs="+",
        default=[DATA_REAL_RAW_EXAMPLE],
        help="Input .pcd path(s), directory(=*.pcd), or glob pattern(s).",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Single output .ply path (only valid when exactly one input is provided).",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=DATA_INTERIM_PCD2PLY_DIR,
        help="Output directory for batch mode (writes <stem>.ply).",
    )
    ap.add_argument("--ascii", action="store_true", help="Write output in ASCII")
    ap.add_argument(
        "--keep_going",
        action="store_true",
        help="Continue converting remaining files even if one fails.",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]

    in_paths = expand_inputs(root, args.in_pcd, exts=[".pcd"])
    if not in_paths:
        raise ValueError("No input .pcd files matched (check path/pattern).")

    if args.out is not None and len(in_paths) != 1:
        raise ValueError("--out can be used only with a single input. Use --out_dir for batch.")

    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Repo root: {root}")
    print(f"[INFO] Matched inputs: {len(in_paths)} file(s)")
    if len(in_paths) <= 10:
        for p in in_paths:
            print(f"       - {p}")
    else:
        print(f"       - {in_paths[0]}")
        print(f"       - ... ({len(in_paths)-2} more) ...")
        print(f"       - {in_paths[-1]}")

    failures: list[tuple[Path, str]] = []

    for in_path in in_paths:
        try:
            if args.out is not None:
                out_path = (root / args.out).resolve()
            else:
                out_path = out_dir / f"{in_path.stem}.ply"

            out_path.parent.mkdir(parents=True, exist_ok=True)

            pcd = read_point_cloud_any(in_path)

            # apply mm->m + axis rotation into PLY frame
            pcd.transform(T_PCD_TO_PLY)

            # quick sanity prints
            pts = np.asarray(pcd.points)
            print(f"[OK] Loaded: {in_path}  N={len(pts)}")
            if len(pts) > 0:
                print(f"     After convert (m, ply-frame): min={pts.min(axis=0)} max={pts.max(axis=0)}")

            ok = o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=args.ascii)
            if not ok:
                raise RuntimeError(f"Failed to write: {out_path}")
            print(f"[OK] Wrote:  {out_path}")

        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            failures.append((in_path, msg))
            print(f"[ERR] {in_path} -> {msg}", file=sys.stderr)
            if not args.keep_going:
                raise

    if failures:
        print("\n[SUMMARY] Failures:", file=sys.stderr)
        for p, msg in failures:
            print(f"  - {p}: {msg}", file=sys.stderr)
        sys.exit(1)

    print("[DONE] All conversions finished successfully.")


if __name__ == "__main__":
    main()

# scripts/pc/center_clouds.py
from __future__ import annotations

import argparse
from pathlib import Path

import open3d as o3d

from scripts.common.io_paths import (
    DATA_INTERIM_CENTERED_REALITY,
    DATA_INTERIM_CENTERED_VIRTUAL,
    DATA_REAL_HUMAN_EXAMPLE,
    DATA_VIRTUAL_POSE324_EXAMPLE,
    resolve_repo_path,
)
from scripts.common.normalize import center_point_cloud_o3d


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Center real/virtual point clouds and write centered outputs.")
    parser.add_argument("--real", type=str, default=DATA_REAL_HUMAN_EXAMPLE, help="Real cloud path (.ply/.pcd)")
    parser.add_argument("--virtual", type=str, default=DATA_VIRTUAL_POSE324_EXAMPLE, help="Virtual cloud path (.ply/.pcd)")
    parser.add_argument("--out-real", type=str, default=DATA_INTERIM_CENTERED_REALITY, help="Centered real output path")
    parser.add_argument("--out-virtual", type=str, default=DATA_INTERIM_CENTERED_VIRTUAL, help="Centered virtual output path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    real_path = resolve_repo_path(repo_root, args.real)
    virtual_path = resolve_repo_path(repo_root, args.virtual)
    out_real = resolve_repo_path(repo_root, args.out_real)
    out_virtual = resolve_repo_path(repo_root, args.out_virtual)

    real_cloud = o3d.io.read_point_cloud(str(real_path))
    virtual_cloud = o3d.io.read_point_cloud(str(virtual_path))

    centered_real, real_offset = center_point_cloud_o3d(real_cloud, method="centroid")
    centered_virtual, virtual_offset = center_point_cloud_o3d(virtual_cloud, method="centroid")

    out_real.parent.mkdir(parents=True, exist_ok=True)
    out_virtual.parent.mkdir(parents=True, exist_ok=True)

    o3d.io.write_point_cloud(str(out_real), centered_real)
    o3d.io.write_point_cloud(str(out_virtual), centered_virtual)

    print("real subtracted:", real_offset)
    print("gt   subtracted:", virtual_offset)


if __name__ == "__main__":
    main()

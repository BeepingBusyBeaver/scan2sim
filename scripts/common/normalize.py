# scripts/common/normalize.py
from __future__ import annotations
import numpy as np
import open3d as o3d

def compute_center_xyz(pts: np.ndarray, method: str = "centroid") -> np.ndarray:
    if method == "centroid":
        return pts.mean(axis=0)
    if method == "median":
        return np.median(pts, axis=0)
    if method == "aabb":
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        return (mn + mx) * 0.5
    raise ValueError(f"Unknown method: {method}")

def center_point_cloud_o3d(
    pcd: o3d.geometry.PointCloud,
    method: str = "centroid",
    center_xy_only: bool = False,
    floor_to_z0: bool = False,
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Returns (pcd_centered, total_subtracted_translation)

    total_subtracted_translation is the vector t such that:
        new_points = old_points - t
    (i.e., we applied Open3D translations of -t in total, with relative=True.)
    """
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return pcd, np.zeros(3, dtype=np.float64)

    c = compute_center_xyz(pts, method=method).astype(np.float64)

    if center_xy_only:
        c = c.copy()
        c[2] = 0.0

    # We'll return the total vector subtracted from the original points.
    t_total = c.copy()

    # Step 1) translate by -c  (new = old - c)
    pcd.translate(-c, relative=True)

    # Step 2) optionally lift the "floor" to z=0
    if floor_to_z0:
        # After Step 1, z values became (old_z - c_z).
        # So min z after centering = min(old_z) - c_z.
        zmin_centered = float(pts[:, 2].min() - c[2])
        pcd.translate([0.0, 0.0, -zmin_centered], relative=True)

        # This second translate subtracts [0,0,zmin_centered] from the original.
        t_total[2] += zmin_centered

    return pcd, t_total

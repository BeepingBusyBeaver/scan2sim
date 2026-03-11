# scripts/parser/types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True)
class PartBox:
    name: str
    valid: bool
    num_points: int
    min_xyz: np.ndarray
    max_xyz: np.ndarray
    center_xyz: np.ndarray
    size_xyz: np.ndarray
    obb_center_xyz: np.ndarray
    obb_size_xyz: np.ndarray
    obb_axes_xyz: np.ndarray
    obb_corners_xyz: np.ndarray

    @staticmethod
    def _axis_aligned_corners(min_xyz: np.ndarray, max_xyz: np.ndarray) -> np.ndarray:
        x0, y0, z0 = [float(v) for v in min_xyz.tolist()]
        x1, y1, z1 = [float(v) for v in max_xyz.tolist()]
        return np.array(
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x1, y1, z0],
                [x0, y1, z0],
                [x0, y0, z1],
                [x1, y0, z1],
                [x1, y1, z1],
                [x0, y1, z1],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _expand_bounds_min_size(min_xyz: np.ndarray, max_xyz: np.ndarray, min_size: float) -> tuple[np.ndarray, np.ndarray]:
        min_arr = np.asarray(min_xyz, dtype=np.float64).copy()
        max_arr = np.asarray(max_xyz, dtype=np.float64).copy()
        if float(min_size) <= 0.0:
            return min_arr, max_arr
        target = float(min_size)
        center = 0.5 * (min_arr + max_arr)
        size = max_arr - min_arr
        for axis in range(3):
            if float(size[axis]) >= target:
                continue
            half = 0.5 * target
            min_arr[axis] = center[axis] - half
            max_arr[axis] = center[axis] + half
        return min_arr, max_arr

    @staticmethod
    def _principal_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        center = np.mean(points, axis=0).astype(np.float64)
        centered = points - center[None, :]
        if points.shape[0] < 3:
            return center, np.eye(3, dtype=np.float64)
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        axes = eigvecs[:, order].astype(np.float64)
        if float(np.linalg.det(axes)) < 0.0:
            axes[:, 2] = -axes[:, 2]
        return center, axes

    @staticmethod
    def _density_inlier_mask(points: np.ndarray, *, k: int, z_thresh: float) -> np.ndarray:
        n_points = int(points.shape[0])
        if n_points <= max(6, int(k) + 1):
            return np.ones((n_points,), dtype=bool)
        diff = points[:, None, :] - points[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        k_eff = max(1, min(int(k), n_points - 1))
        nn = np.partition(dist, kth=k_eff - 1, axis=1)[:, :k_eff]
        local = np.mean(nn, axis=1)
        med = float(np.median(local))
        mad = float(np.median(np.abs(local - med)))
        if mad <= 1e-9:
            return np.ones((n_points,), dtype=bool)
        scale = 1.4826 * mad
        score = (local - med) / max(scale, 1e-9)
        return score <= float(z_thresh)

    @staticmethod
    def _largest_component_mask(points: np.ndarray, *, radius: float) -> np.ndarray:
        n_points = int(points.shape[0])
        if n_points <= 2:
            return np.ones((n_points,), dtype=bool)
        diff = points[:, None, :] - points[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        adjacency = dist <= max(float(radius), 1e-6)
        np.fill_diagonal(adjacency, False)

        visited = np.zeros((n_points,), dtype=bool)
        best_component: list[int] = []
        for start in range(n_points):
            if visited[start]:
                continue
            stack = [start]
            component: list[int] = []
            visited[start] = True
            while stack:
                node = stack.pop()
                component.append(node)
                neighbors = np.flatnonzero(adjacency[node])
                for neigh in neighbors.tolist():
                    if visited[neigh]:
                        continue
                    visited[neigh] = True
                    stack.append(int(neigh))
            if len(component) > len(best_component):
                best_component = component
        mask = np.zeros((n_points,), dtype=bool)
        if best_component:
            mask[np.array(best_component, dtype=np.int64)] = True
        return mask

    @staticmethod
    def _robust_fit_points(points: np.ndarray, *, min_points: int, robust_quantile: float) -> np.ndarray:
        n_points = int(points.shape[0])
        if n_points <= max(3, int(min_points)):
            return points
        keep_min = max(int(min_points), int(np.ceil(n_points * 0.30)))

        filtered = points
        density_mask = PartBox._density_inlier_mask(filtered, k=6, z_thresh=2.8)
        if int(np.count_nonzero(density_mask)) >= keep_min:
            filtered = filtered[density_mask]

        n_filtered = int(filtered.shape[0])
        if n_filtered >= max(6, int(min_points) + 2):
            diff = filtered[:, None, :] - filtered[None, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, np.inf)
            k_eff = max(1, min(6, n_filtered - 1))
            kth = np.partition(dist, kth=k_eff - 1, axis=1)[:, k_eff - 1]
            radius = max(float(np.median(kth)) * 1.8, 0.018)
            component_mask = PartBox._largest_component_mask(filtered, radius=radius)
            if int(np.count_nonzero(component_mask)) >= keep_min:
                filtered = filtered[component_mask]

        q = float(np.clip(robust_quantile, 0.0, 0.20))
        if q <= 0.0:
            return filtered
        n_filtered = int(filtered.shape[0])
        if n_filtered < max(24, int(min_points) * 2):
            return filtered
        center, axes = PartBox._principal_axes(filtered)
        proj = (filtered - center[None, :]) @ axes
        lo = np.quantile(proj, q, axis=0)
        hi = np.quantile(proj, 1.0 - q, axis=0)
        keep = np.all((proj >= lo[None, :]) & (proj <= hi[None, :]), axis=1)
        q_filtered = filtered[keep]
        min_keep = max(int(min_points), int(np.ceil(n_filtered * 0.45)))
        if q_filtered.shape[0] < min_keep:
            return filtered
        return q_filtered

    @staticmethod
    def _fit_obb(points: np.ndarray, *, min_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        base_center, axes = PartBox._principal_axes(points)
        local = (points - base_center[None, :]) @ axes
        min_local = np.min(local, axis=0).astype(np.float64)
        max_local = np.max(local, axis=0).astype(np.float64)
        min_local, max_local = PartBox._expand_bounds_min_size(min_local, max_local, min_size=min_size)
        extent = (max_local - min_local).astype(np.float64)
        center_local = 0.5 * (min_local + max_local)
        obb_center = (base_center + axes @ center_local).astype(np.float64)
        x0, y0, z0 = [float(v) for v in min_local.tolist()]
        x1, y1, z1 = [float(v) for v in max_local.tolist()]
        corners_local = np.array(
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x1, y1, z0],
                [x0, y1, z0],
                [x0, y0, z1],
                [x1, y0, z1],
                [x1, y1, z1],
                [x0, y1, z1],
            ],
            dtype=np.float64,
        )
        corners = corners_local @ axes.T + obb_center[None, :]
        return obb_center, extent, axes, corners

    @staticmethod
    def empty(name: str) -> "PartBox":
        zeros = np.zeros((3,), dtype=np.float64)
        axes = np.eye(3, dtype=np.float64)
        corners = np.zeros((8, 3), dtype=np.float64)
        return PartBox(
            name=name,
            valid=False,
            num_points=0,
            min_xyz=zeros.copy(),
            max_xyz=zeros.copy(),
            center_xyz=zeros.copy(),
            size_xyz=zeros.copy(),
            obb_center_xyz=zeros.copy(),
            obb_size_xyz=zeros.copy(),
            obb_axes_xyz=axes,
            obb_corners_xyz=corners,
        )

    @staticmethod
    def from_points(
        name: str,
        points: np.ndarray,
        min_points: int,
        *,
        allow_small: bool = False,
        min_size: float = 0.0,
        robust_quantile: float = 0.03,
    ) -> "PartBox":
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Part '{name}' points must be Nx3.")
        count = int(points.shape[0])
        if count <= 0:
            return PartBox.empty(name)
        if count < int(min_points) and not bool(allow_small):
            return PartBox.empty(name)

        fit_points = PartBox._robust_fit_points(
            points.astype(np.float64),
            min_points=int(min_points),
            robust_quantile=robust_quantile,
        )
        min_xyz = np.min(fit_points, axis=0).astype(np.float64)
        max_xyz = np.max(fit_points, axis=0).astype(np.float64)
        min_xyz, max_xyz = PartBox._expand_bounds_min_size(min_xyz, max_xyz, min_size=min_size)
        center_xyz = ((min_xyz + max_xyz) * 0.5).astype(np.float64)
        size_xyz = (max_xyz - min_xyz).astype(np.float64)
        obb_center_xyz, obb_size_xyz, obb_axes_xyz, obb_corners_xyz = PartBox._fit_obb(
            fit_points,
            min_size=min_size,
        )
        return PartBox(
            name=name,
            valid=(count >= int(min_points)) or bool(allow_small),
            num_points=count,
            min_xyz=min_xyz,
            max_xyz=max_xyz,
            center_xyz=center_xyz,
            size_xyz=size_xyz,
            obb_center_xyz=obb_center_xyz,
            obb_size_xyz=obb_size_xyz,
            obb_axes_xyz=obb_axes_xyz,
            obb_corners_xyz=obb_corners_xyz,
        )

    @staticmethod
    def from_bounds(
        name: str,
        *,
        num_points: int,
        min_xyz: np.ndarray,
        max_xyz: np.ndarray,
        template: "PartBox | None" = None,
    ) -> "PartBox":
        min_arr = np.asarray(min_xyz, dtype=np.float64).copy()
        max_arr = np.asarray(max_xyz, dtype=np.float64).copy()
        center = ((min_arr + max_arr) * 0.5).astype(np.float64)
        size = (max_arr - min_arr).astype(np.float64)
        if template is not None and bool(getattr(template, "valid", False)):
            delta = center - np.asarray(template.center_xyz, dtype=np.float64)
            obb_center = np.asarray(template.obb_center_xyz, dtype=np.float64) + delta
            obb_size = np.asarray(template.obb_size_xyz, dtype=np.float64).copy()
            obb_axes = np.asarray(template.obb_axes_xyz, dtype=np.float64).copy()
            obb_corners = np.asarray(template.obb_corners_xyz, dtype=np.float64) + delta[None, :]
        else:
            obb_center = center.copy()
            obb_size = size.copy()
            obb_axes = np.eye(3, dtype=np.float64)
            obb_corners = PartBox._axis_aligned_corners(min_arr, max_arr)
        return PartBox(
            name=name,
            valid=True,
            num_points=int(num_points),
            min_xyz=min_arr,
            max_xyz=max_arr,
            center_xyz=center,
            size_xyz=size,
            obb_center_xyz=obb_center,
            obb_size_xyz=obb_size,
            obb_axes_xyz=obb_axes,
            obb_corners_xyz=obb_corners,
        )

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "valid": self.valid,
            "num_points": self.num_points,
            "min_xyz": [float(v) for v in self.min_xyz.tolist()],
            "max_xyz": [float(v) for v in self.max_xyz.tolist()],
            "center_xyz": [float(v) for v in self.center_xyz.tolist()],
            "size_xyz": [float(v) for v in self.size_xyz.tolist()],
            "obb_center_xyz": [float(v) for v in self.obb_center_xyz.tolist()],
            "obb_size_xyz": [float(v) for v in self.obb_size_xyz.tolist()],
            "obb_axes_xyz": [[float(v) for v in row] for row in self.obb_axes_xyz.tolist()],
            "obb_corners_xyz": [[float(v) for v in row] for row in self.obb_corners_xyz.tolist()],
        }

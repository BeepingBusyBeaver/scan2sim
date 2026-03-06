from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.feature.heads import HeadSpec


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    # src: [B, N, 3], dst: [B, M, 3] -> dist: [B, N, M]
    return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    # points: [B, N, C], idx: [B, S] or [B, S, K]
    device = points.device
    batch_size = points.shape[0]
    view_shape = [batch_size] + [1] * (idx.dim() - 1)
    repeat_shape = [1] + list(idx.shape[1:])
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    # xyz: [B, N, 3] -> centroids idx: [B, npoint]
    device = xyz.device
    batch_size, num_points, _ = xyz.shape
    npoint = max(1, int(npoint))

    centroids = torch.zeros((batch_size, npoint), dtype=torch.long, device=device)
    distance = torch.full((batch_size, num_points), 1e10, device=device)
    farthest = torch.randint(0, max(num_points, 1), (batch_size,), dtype=torch.long, device=device)
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch_size, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def knn_point(k: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    # xyz: [B, N, 3], new_xyz: [B, S, 3] -> idx: [B, S, k]
    dist = square_distance(new_xyz, xyz)
    num_points = xyz.shape[1]
    k_use = min(max(int(k), 1), max(num_points, 1))
    idx = torch.topk(dist, k=k_use, dim=-1, largest=False, sorted=False)[1]

    if k_use == k:
        return idx

    pad = idx[..., -1:].expand(*idx.shape[:-1], k - k_use)
    return torch.cat([idx, pad], dim=-1)


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint: int, k: int, in_channels: int, mlp_channels: Sequence[int]) -> None:
        super().__init__()
        self.npoint = int(npoint)
        self.k = int(k)

        last_channels = in_channels + 3
        layers: List[nn.Module] = []
        for out_channels in mlp_channels:
            layers.append(nn.Conv2d(last_channels, int(out_channels), kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(int(out_channels)))
            layers.append(nn.ReLU(inplace=True))
            last_channels = int(out_channels)
        self.mlp = nn.Sequential(*layers)
        self.out_channels = last_channels

    def forward(self, xyz: torch.Tensor, features: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        # xyz: [B, N, 3], features: [B, C, N] or None
        fps_idx = farthest_point_sample(xyz, self.npoint)  # [B, S]
        new_xyz = index_points(xyz, fps_idx)  # [B, S, 3]

        group_idx = knn_point(self.k, xyz, new_xyz)  # [B, S, K]
        grouped_xyz = index_points(xyz, group_idx)  # [B, S, K, 3]
        grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)  # [B, S, K, 3]

        if features is None:
            grouped = grouped_xyz_norm
        else:
            features_t = features.transpose(1, 2).contiguous()  # [B, N, C]
            grouped_features = index_points(features_t, group_idx)  # [B, S, K, C]
            grouped = torch.cat([grouped_xyz_norm, grouped_features], dim=-1)  # [B, S, K, C+3]

        grouped = grouped.permute(0, 3, 2, 1).contiguous()  # [B, C+3, K, S]
        grouped = self.mlp(grouped)  # [B, C_out, K, S]
        new_features = torch.max(grouped, dim=2)[0]  # [B, C_out, S]
        return new_xyz, new_features


class PointNetPPEncoder(nn.Module):
    def __init__(
        self,
        *,
        sa_npoints: Sequence[int],
        sa_k: Sequence[int],
        sa_mlps: Sequence[Sequence[int]],
        global_mlp: Sequence[int],
    ) -> None:
        super().__init__()
        if not (len(sa_npoints) == len(sa_k) == len(sa_mlps)):
            raise ValueError("sa_npoints, sa_k, sa_mlps must have same length")
        if not sa_npoints:
            raise ValueError("At least one set abstraction stage is required.")

        self.sa_layers = nn.ModuleList()
        last_channels = 0
        for npoint, k, mlp in zip(sa_npoints, sa_k, sa_mlps):
            if not mlp:
                raise ValueError("Each sa_mlps entry must be non-empty.")
            stage = PointNetSetAbstraction(
                npoint=int(npoint),
                k=int(k),
                in_channels=int(last_channels),
                mlp_channels=[int(ch) for ch in mlp],
            )
            self.sa_layers.append(stage)
            last_channels = stage.out_channels

        global_layers: List[nn.Module] = []
        global_in = int(last_channels)
        for hidden in global_mlp:
            hidden = int(hidden)
            global_layers.append(nn.Linear(global_in, hidden, bias=False))
            global_layers.append(nn.BatchNorm1d(hidden))
            global_layers.append(nn.ReLU(inplace=True))
            global_in = hidden
        self.global_mlp = nn.Sequential(*global_layers)
        self.output_dim = global_in

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # points: [B, N, 3]
        xyz = points
        features = None
        for stage in self.sa_layers:
            xyz, features = stage(xyz, features)
        assert features is not None
        pooled = torch.max(features, dim=-1)[0]  # [B, C]
        return self.global_mlp(pooled)


@dataclass(frozen=True)
class MultiHeadLayout:
    head_names: List[str]
    head_num_classes: List[int]


class PointNet2MultiHeadClassifier(nn.Module):
    def __init__(
        self,
        *,
        encoder: PointNetPPEncoder,
        layout: MultiHeadLayout,
        head_hidden: int,
        head_dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.layout = layout

        hidden = int(head_hidden)
        self.shared_head = nn.Sequential(
            nn.Linear(encoder.output_dim, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(head_dropout)),
        )
        self.head_layers = nn.ModuleDict(
            {
                name: nn.Linear(hidden, int(num_classes))
                for name, num_classes in zip(layout.head_names, layout.head_num_classes)
            }
        )

    def forward(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.encoder(points)
        shared = self.shared_head(features)
        return {name: head_layer(shared) for name, head_layer in self.head_layers.items()}


def build_model_from_config(
    model_cfg: Mapping[str, object],
    heads: Sequence[HeadSpec],
) -> PointNet2MultiHeadClassifier:
    encoder = PointNetPPEncoder(
        sa_npoints=model_cfg.get("sa_npoints", [512, 128]),
        sa_k=model_cfg.get("sa_k", [32, 32]),
        sa_mlps=model_cfg.get("sa_mlps", [[64, 64, 128], [128, 128, 256]]),
        global_mlp=model_cfg.get("global_mlp", [256, 512]),
    )
    layout = MultiHeadLayout(
        head_names=[head.name for head in heads],
        head_num_classes=[len(head.class_values) for head in heads],
    )
    return PointNet2MultiHeadClassifier(
        encoder=encoder,
        layout=layout,
        head_hidden=int(model_cfg.get("head_hidden", 256)),
        head_dropout=float(model_cfg.get("head_dropout", 0.2)),
    )


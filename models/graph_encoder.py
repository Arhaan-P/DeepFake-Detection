"""
Skeleton-Graph Encoder (future work, plan Section 7 / RQ9)
==========================================================
Spatial-temporal graph convolution (ST-GCN, Yan et al. 2018; the backbone
family of GaitGraph) over the 12 gait joints. RQ9 asks whether modelling
explicit body topology beats treating the 78-D descriptor as a generic time
series, so this encoder consumes exactly the baseline channels, re-arranged
per joint:

    node features = [x, y, z, vx, vy, vz, joint angle]   (7 per joint)

where the angle channel is the knee/hip/ankle flexion angle for those six
joints and zero for shoulders, heels and toes.

Output keeps the time axis, (B, T, out_dim), so it drops into the same
encode-then-difference verifier as every other encoder
(models/verifier_variants.py).

Author: DeepFake Detection Project
"""

from typing import Dict, List, Tuple

import torch
from torch import nn

# 12-joint gait layout: 0 LSh, 1 RSh, 2 LHip, 3 RHip, 4 LKnee, 5 RKnee,
# 6 LAnk, 7 RAnk, 8 LHeel, 9 RHeel, 10 LFoot, 11 RFoot
GAIT_EDGES: List[Tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (6, 10),
    (7, 11),
    (8, 10),
    (9, 11),
]
# baseline angle channel order -> joint it belongs to
ANGLE_TO_JOINT = [4, 5, 2, 3, 6, 7]  # L/R knee, L/R hip, L/R ankle
NUM_JOINTS = 12
NODE_CHANNELS = 7


def adjacency_partitions(num_joints: int = NUM_JOINTS, edges=GAIT_EDGES):
    """(2, V, V): identity, and the symmetrically normalised neighbour graph."""
    a = torch.zeros(num_joints, num_joints)
    for i, j in edges:
        a[i, j] = a[j, i] = 1.0
    deg = a.sum(1).clamp(min=1)
    norm = a / torch.sqrt(deg[:, None] * deg[None, :])
    return torch.stack([torch.eye(num_joints), norm])


class STGCNBlock(nn.Module):
    """Graph conv over joints, then temporal conv over frames, residual."""

    def __init__(self, cin, cout, partitions, t_kernel=9, dropout=0.1):
        super().__init__()
        k = partitions.shape[0]
        self.register_buffer("A", partitions)
        self.edge_importance = nn.Parameter(torch.ones_like(partitions))
        self.gcn = nn.Conv2d(cin, cout * k, kernel_size=1)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(cout),
            nn.ReLU(),
            nn.Conv2d(cout, cout, (t_kernel, 1), padding=(t_kernel // 2, 0)),
            nn.BatchNorm2d(cout),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Identity()
            if cin == cout
            else nn.Sequential(nn.Conv2d(cin, cout, 1), nn.BatchNorm2d(cout))
        )
        self.k, self.cout = k, cout

    def forward(self, x):  # x: (B, C, T, V)
        b, _, t, v = x.shape
        y = self.gcn(x).view(b, self.k, self.cout, t, v)
        y = torch.einsum("bkctv,kvw->bctw", y, self.A * self.edge_importance)
        return torch.relu(self.tcn(y) + self.residual(x))


class STGCNEncoder(nn.Module):
    """(B, T, D) descriptor -> (B, T, out_dim) via ST-GCN over 12 joints.

    Args:
        slices: channel ranges of the descriptor's families
            (utils.gait_features.family_slices); needs `coords`, uses
            `velocity` and `angles` when present.
    """

    def __init__(
        self,
        slices: Dict[str, slice],
        channels=(32, 64, 64),
        out_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        if "coords" not in slices:
            raise ValueError("STGCNEncoder needs the 'coords' family")
        self.slices = slices
        parts = adjacency_partitions()
        self.data_bn = nn.BatchNorm1d(NODE_CHANNELS * NUM_JOINTS)
        blocks, cin = [], NODE_CHANNELS
        for c in channels:
            blocks.append(STGCNBlock(cin, c, parts, dropout=dropout))
            cin = c
        self.blocks = nn.ModuleList(blocks)
        self.proj = nn.Linear(cin, out_dim)
        self.out_dim = out_dim

    def to_nodes(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, 7, T, 12) node tensor."""
        b, t, _ = x.shape
        coords = x[..., self.slices["coords"]].reshape(b, t, NUM_JOINTS, 3)
        if "velocity" in self.slices:
            vel = x[..., self.slices["velocity"]].reshape(b, t, NUM_JOINTS, 3)
        else:
            vel = torch.zeros_like(coords)
        ang = x.new_zeros(b, t, NUM_JOINTS, 1)
        if "angles" in self.slices:
            a = x[..., self.slices["angles"]]
            ang[:, :, ANGLE_TO_JOINT, 0] = a
        nodes = torch.cat([coords, vel, ang], dim=-1)  # (B, T, V, 7)
        return nodes.permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        n = self.to_nodes(x)  # (B, C, T, V)
        b, c, t, v = n.shape
        n = self.data_bn(n.permute(0, 3, 1, 2).reshape(b, v * c, t))
        n = n.reshape(b, v, c, t).permute(0, 2, 3, 1)
        for blk in self.blocks:
            n = blk(n)
        return self.proj(n.mean(dim=3).transpose(1, 2))  # (B, T, out_dim)

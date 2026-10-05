"""
Verifier Variants for Controlled Model Experiments (plan Section 7 / E7)
=======================================================================
Every variant maps (video V, claimed signature C), both (B, T, D), to two
logits [deepfake, authentic], so all of them train and evaluate through the
same harness (scripts/future_work/run_experiment.py) on the same folds.

  tcn               The deployed decision path: [V-C, |V-C|, V*C] -> temporal
                    CNN (k = 7, 5, 3) -> MLP. With D = 78 and hidden 64 this
                    is exactly models/full_pipeline.py's diff_conv +
                    diff_classifier, 133,058 parameters.
  raw+<enc>         Encode V and C with shared weights (time preserved), and
                    compare BOTH the raw and the encoded sequences, as in the
                    definitive ablation (scripts/evaluation/ablation_loocv.py):
                    an encoder must add something on top of the deployed path.
  <enc>_only        Encoded comparison only (can an encoder replace raw
                    differencing?).
  raw+freq          Deployed path + a frequency branch comparing per-channel
                    spectral magnitudes of V and C (RQ5).
  siamese           Metric learning: one shared temporal-CNN embedding per
                    sequence, score = learned affine map of cosine similarity
                    (the "compare against the difference-based verifier" row).

Encoders: bilstm, transformer, cnn (GaitEncoder), hybrid (GaitEncoder ->
BiLSTM || Transformer), stgcn (models/graph_encoder.py, RQ9).

The non-learned DTW template matcher lives in the harness, since it has no
parameters to train.

Author: DeepFake Detection Project
"""

from typing import Dict, Optional

import torch
from torch import nn


def _init(module: nn.Module) -> None:
    """Same initialisation as GaitDeepfakeDetector._init_weights."""
    for m in module.modules():
        if isinstance(m, nn.LSTM):
            continue
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def compare(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = a - b
    return torch.cat([d, torch.abs(d), a * b], dim=2)


class DiffHead(nn.Module):
    """Temporal CNN over comparison features + MLP (the deployed head)."""

    def __init__(self, comparison_dim: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(comparison_dim, hidden, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden // 2, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )
        self.pooled_dim = hidden // 2

    def pooled(self, f: torch.Tensor) -> torch.Tensor:
        return self.conv(f.permute(0, 2, 1)).squeeze(-1)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pooled(f))


class TCNVerifier(nn.Module):
    """The deployed raw-difference verifier."""

    def __init__(self, input_dim: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.head = DiffHead(3 * input_dim, hidden, dropout)
        _init(self)

    def forward(self, v, c):
        return self.head(compare(v, c))


class EncodedDiffVerifier(nn.Module):
    """Encode-then-difference, optionally fused with the raw comparison."""

    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int,
        fuse_raw: bool = True,
        hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.fuse_raw = fuse_raw
        dim = 3 * encoder.out_dim + (3 * input_dim if fuse_raw else 0)
        self.head = DiffHead(dim, hidden, dropout)
        _init(self)

    def forward(self, v, c):
        n = v.size(0)
        # one batched pass so BatchNorm sees V and C together (see
        # ablation_loocv.py for why this matters)
        h = self.encoder(torch.cat([v, c], dim=0))
        parts = [compare(h[:n], h[n:])]
        if self.fuse_raw:
            parts.insert(0, compare(v, c))
        return self.head(torch.cat(parts, dim=2))


class FrequencyFusionVerifier(nn.Module):
    """Deployed path + spectral comparison branch (RQ5).

    The branch takes |rFFT| over time of every channel (first `n_bins`
    bins), compares V and C spectra as [|Fv - Fc|, Fv * Fc], and projects to
    the head's pooled width; both pooled vectors feed one classifier.
    """

    def __init__(
        self, input_dim: int, n_bins: int = 12, hidden: int = 64, dropout: float = 0.1
    ):
        super().__init__()
        self.n_bins = n_bins
        self.head = DiffHead(3 * input_dim, hidden, dropout)
        self.freq = nn.Sequential(
            nn.Linear(2 * n_bins * input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.head.pooled_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.head.pooled_dim, self.head.pooled_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.head.pooled_dim, 2),
        )
        _init(self)

    def spectrum(self, x: torch.Tensor) -> torch.Tensor:
        mag = torch.abs(torch.fft.rfft(x - x.mean(dim=1, keepdim=True), dim=1))
        return torch.log1p(mag[:, 1 : self.n_bins + 1])  # drop DC

    def forward(self, v, c):
        fv, fc = self.spectrum(v), self.spectrum(c)
        spec = torch.cat([torch.abs(fv - fc), fv * fc], dim=1).flatten(1)
        z = torch.cat([self.head.pooled(compare(v, c)), self.freq(spec)], dim=1)
        return self.classifier(z)


class SiameseVerifier(nn.Module):
    """Shared embedding per sequence; logit = s * cos(e_v, e_c) + b."""

    def __init__(
        self, input_dim: int, hidden: int = 64, emb: int = 64, dropout: float = 0.1
    ):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(input_dim, hidden, 7, padding=3),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(hidden, emb)
        self.scale = nn.Parameter(torch.tensor(5.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
        _init(self)

    def embed(self, x):
        z = self.proj(self.enc(x.permute(0, 2, 1)).squeeze(-1))
        return nn.functional.normalize(z, dim=1)

    def forward(self, v, c):
        n = v.size(0)
        e = self.embed(torch.cat([v, c], dim=0))
        cos = (e[:n] * e[n:]).sum(dim=1)
        z = self.scale * cos + self.bias
        return torch.stack([torch.zeros_like(z), z], dim=1)


# ------------------------------------------------------------
# encoders, each (B, T, D) -> (B, T, out_dim)
# ------------------------------------------------------------


class _SeqWrap(nn.Module):
    """Adapts the project's (outputs, summary)-returning encoders."""

    def __init__(self, enc, out_dim):
        super().__init__()
        self.enc, self.out_dim = enc, out_dim

    def forward(self, x):
        out = self.enc(x)
        return out[0] if isinstance(out, tuple) else out


def build_encoder(
    kind: str, input_dim: int, slices: Optional[Dict] = None, dropout: float = 0.1
):
    if kind == "bilstm":
        from models.temporal_model import BiLSTMEncoder

        enc = BiLSTMEncoder(
            input_dim=input_dim, hidden_dim=64, num_layers=1, dropout=dropout
        )
        return _SeqWrap(enc, enc.output_dim)
    if kind == "transformer":
        from models.temporal_model import TransformerEncoder

        enc = TransformerEncoder(
            input_dim=input_dim, d_model=128, nhead=4, num_layers=2, dropout=dropout
        )
        return _SeqWrap(enc, 128)
    if kind == "cnn":
        from models.gait_encoder import GaitEncoder

        enc = GaitEncoder(
            input_dim=input_dim, hidden_dims=(64, 128), output_dim=128, dropout=dropout
        )
        return _SeqWrap(enc, 128)
    if kind == "hybrid":
        from models.gait_encoder import GaitEncoder
        from models.temporal_model import DualPathTemporalModel

        seq = nn.Sequential(
            GaitEncoder(input_dim, (64, 128), 128, dropout=dropout),
        )
        temporal = DualPathTemporalModel(
            input_dim=128,
            lstm_hidden=64,
            lstm_layers=1,
            transformer_d_model=128,
            transformer_heads=4,
            transformer_layers=2,
            output_dim=128,
            dropout=dropout,
        )

        class _Hybrid(nn.Module):
            out_dim = 128

            def __init__(self):
                super().__init__()
                self.cnn, self.temporal = seq, temporal

            def forward(self, x):
                return self.temporal(self.cnn(x))[0]

        return _Hybrid()
    if kind == "stgcn":
        from models.graph_encoder import STGCNEncoder

        return STGCNEncoder(slices, dropout=dropout)
    raise ValueError(f"Unknown encoder '{kind}'")


VERIFIERS = [
    "tcn",
    "raw+bilstm",
    "raw+transformer",
    "raw+cnn",
    "raw+hybrid",
    "raw+stgcn",
    "bilstm_only",
    "transformer_only",
    "stgcn_only",
    "raw+freq",
    "siamese",
]


def build_verifier(
    name: str,
    input_dim: int,
    slices: Optional[Dict] = None,
    dropout: float = 0.1,
    hidden: int = 64,
) -> nn.Module:
    if name == "tcn":
        return TCNVerifier(input_dim, hidden, dropout)
    if name == "raw+freq":
        return FrequencyFusionVerifier(input_dim, hidden=hidden, dropout=dropout)
    if name == "siamese":
        return SiameseVerifier(input_dim, hidden=hidden, dropout=dropout)
    if name.startswith("raw+"):
        enc = build_encoder(name[4:], input_dim, slices, dropout)
        return EncodedDiffVerifier(enc, input_dim, True, hidden, dropout)
    if name.endswith("_only"):
        enc = build_encoder(name[: -len("_only")], input_dim, slices, dropout)
        return EncodedDiffVerifier(enc, input_dim, False, hidden, dropout)
    raise ValueError(f"Unknown verifier '{name}'. Options: {VERIFIERS + ['dtw']}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

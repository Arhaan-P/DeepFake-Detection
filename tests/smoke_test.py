"""
Pipeline smoke test.
====================
Verifies the model builds and runs end-to-end on synthetic data, without a
dataset, GPU, or trained checkpoint. This is a coverage floor ("does it
crash"), not a correctness test — it asserts output shapes only, and makes
no assertion about which internal branch the verification-mode decision
actually flows through (see AUDIT_FINDINGS.md for that).

Usage:
    python tests/smoke_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models.full_pipeline import create_model


def main() -> int:
    device = torch.device("cpu")
    batch_size, seq_len, input_dim = 2, 60, 78

    model = create_model().to(device)
    model.eval()

    video_features = torch.randn(batch_size, seq_len, input_dim, device=device)
    claimed_features = torch.randn(batch_size, seq_len, input_dim, device=device)

    with torch.no_grad():
        verification_out = model(video_features, claimed_features, mode="verification")
        classification_out = model(video_features, mode="classification")
        embedding_out = model(video_features, mode="embedding")

    assert verification_out["is_authentic"].shape == (batch_size,)
    assert verification_out["similarity"].shape == (batch_size,)
    assert verification_out["confidence"].shape == (batch_size,)
    assert verification_out["video_embedding"].shape == (
        batch_size,
        model.embedding_dim,
    )

    assert classification_out["logits"].shape == (batch_size, 2)
    assert classification_out["prediction"].shape == (batch_size,)

    assert embedding_out["video_embedding"].shape == (batch_size, model.embedding_dim)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model built: {total_params:,} parameters")
    print(f"Verification output shapes OK: {list(verification_out.keys())}")
    print(f"Classification output shapes OK: {list(classification_out.keys())}")
    print(f"Embedding output shapes OK: {list(embedding_out.keys())}")
    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

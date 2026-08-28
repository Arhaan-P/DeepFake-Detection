"""
Pre-flight check for the corrected ablation study.

Asserts the property the fix exists to establish: that each variant's named
branch (CNN / BiLSTM / Transformer) actually receives gradient from the
cross-entropy objective, i.e. that it is genuinely on the decision path.

Under the original script every one of these assertions would fail, because
the branch produced an embedding that never reached the logits.

Run:  venv/Scripts/python.exe scripts/evaluation/_verify_ablation_fix.py
"""

import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# The ablation module imports the data loader at module scope, which drags in
# the whole MediaPipe/TF stack. We only need the nn.Module definitions here.
for mod, attrs in (("utils.data_loader", ["create_data_loaders"]),
                   ("utils.logger", ["setup_logging", "close_logging"])):
    if mod not in sys.modules:
        stub = types.ModuleType(mod)
        for a in attrs:
            setattr(stub, a, lambda *args, **kw: None)
        sys.modules[mod] = stub

import torch
import torch.nn as nn

from scripts.evaluation.ablation_study import (
    CNNOnlyModel, LSTMOnlyModel, TransformerOnlyModel, FullHybridModel)

BUILDERS = {
    "CNN-Only": lambda: CNNOnlyModel(
        input_dim=78, hidden_dims=(64, 128), output_dim=128,
        verification_hidden=64, dropout=0.1),
    "LSTM-Only": lambda: LSTMOnlyModel(
        input_dim=78, lstm_hidden=64, lstm_layers=1,
        verification_hidden=64, dropout=0.1),
    "Transformer-Only": lambda: TransformerOnlyModel(
        input_dim=78, d_model=128, nhead=4, num_layers=2,
        verification_hidden=64, dropout=0.1),
    "Full Hybrid": lambda: FullHybridModel(
        input_dim=78, encoder_hidden_dims=(64, 128), encoder_output_dim=128,
        lstm_hidden=64, lstm_layers=1, transformer_d_model=128,
        transformer_heads=4, transformer_layers=2, embedding_dim=128,
        verification_hidden=64, dropout=0.1),
}

# Which submodule is "the branch under test" for each variant.
BRANCH_PREFIX = {
    "CNN-Only": ("encoder",),
    "LSTM-Only": ("bilstm",),
    "Transformer-Only": ("transformer",),
    "Full Hybrid": ("encoder", "temporal"),
}

B, T, D = 4, 60, 78
failures = []

print("=" * 74)
print("  Pre-flight: does each branch receive gradient from the CE loss?")
print("=" * 74)

for name, build in BUILDERS.items():
    torch.manual_seed(0)
    model = build()
    model.train()

    video = torch.randn(B, T, D)
    claimed = torch.randn(B, T, D)
    labels = torch.tensor([1, 0, 1, 0])

    out = model(video, claimed, mode="verification")
    logits = out["verification"]["logits"]

    assert logits.shape == (B, 2), f"{name}: bad logits shape {logits.shape}"

    loss = nn.CrossEntropyLoss()(logits, labels)
    model.zero_grad()
    loss.backward()

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Parameters carrying a non-zero gradient are, by definition, on the path
    # from input to loss.
    on_path = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None and p.grad.abs().sum().item() > 0:
            on_path += p.numel()

    # Specifically check the named branch, not just the head.
    branch_params, branch_grad = 0, 0
    for mod_name in BRANCH_PREFIX[name]:
        branch = getattr(model, mod_name)
        for p in branch.parameters():
            if not p.requires_grad:
                continue
            branch_params += p.numel()
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                branch_grad += p.numel()

    pct_path = 100.0 * on_path / total
    pct_branch = 100.0 * branch_grad / branch_params if branch_params else 0.0

    ok = branch_grad > 0
    status = "PASS" if ok else "FAIL"
    if not ok:
        failures.append(name)

    print(f"\n  {name}")
    print(f"    total trainable      {total:>9,d}")
    print(f"    receiving gradient   {on_path:>9,d}  ({pct_path:5.1f}%)")
    print(f"    branch params        {branch_params:>9,d}")
    print(f"    branch w/ gradient   {branch_grad:>9,d}  ({pct_branch:5.1f}%)  [{status}]")

print("\n" + "=" * 74)
if failures:
    print("  FAILED -- branch receives no gradient in: " + ", ".join(failures))
    raise SystemExit(1)
print("  All four variants: named branch is on the decision path.")
print("=" * 74)

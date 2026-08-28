"""
Ablation Study for Gait-Based Deepfake Detection
=================================================
Trains and evaluates 4 model variants to measure each component's contribution
to VERIFICATION DISCRIMINATIVENESS:

1. CNN-only:         GaitEncoder embedding only (no temporal modeling)
2. LSTM-only:        BiLSTM embedding only (no CNN, no Transformer)
3. Transformer-only: Transformer embedding only (no CNN, no LSTM)
4. Full Hybrid:      CNN -> (BiLSTM + Transformer) fused embedding

CORRECTED VERSION (see NOTES.md Section 2.2 in the paper repo). The original
version of this script computed every variant's verification logits from an
identical `diff_conv` -> `diff_classifier` head applied to the RAW 78-dim
(video - claimed) feature difference. Each variant's named branch (CNN /
LSTM / Transformer) produced an embedding that was returned in the output
dict but never reached the logits and received no gradient from the
cross-entropy loss -- so all four variants were, numerically, the same
classifier trained four times, and the accuracy spread between them measured
training variance, not component contribution.

This version fixes that: each variant encodes BOTH the video sequence and the
claimed-identity sequence through its OWN branch (shared weights between the
two forward passes, i.e. Siamese), then classifies a comparison of the two
resulting embeddings:

    combined = [e_v || e_c || |e_v - e_c| || e_v * e_c]
    logits   = MLP(combined)

so the branch under test is now the only thing standing between the input and
the decision -- exactly what an ablation over "which temporal branch" is
supposed to isolate. All four variants share this same comparison-and-classify
head structure (not the same instantiated weights); what differs is only the
encoder that produces e_v and e_c.

SUPERSEDED (see NOTES.md Section 2.2). Mean-pooling each sequence to a single
embedding before comparing (as this version does) discards the per-timestep
alignment between the observed sequence and the enrolled signature, which is
the signal the deployed model actually uses -- every variant here lands at
40-79% AUC, which measures how hard metric learning is on this cohort size,
not which encoder helps. `ablation_loocv.py` is the definitive replacement:
it keeps the time axis and runs paired 13-fold x 3-seed LOOCV. This script is
kept for provenance only; do not use its numbers.

Usage:
    python scripts/evaluation/ablation_study.py --features_file data/gait_features/gait_features.pkl --epochs 50

Author: DeepFake Detection Project
"""

import argparse
import json
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.data_loader import create_data_loaders
from utils.logger import close_logging, setup_logging

# ============================================================
# Ablation Model Variants
# ============================================================
#
# Shared comparison-and-classify head. Every variant below builds its own
# instance of this class -- it is NOT shared weights across variants -- and
# is fed embeddings from that variant's own encoder branch, applied
# identically (shared weights) to the video and the claimed sequences.


class EmbeddingComparisonHead(nn.Module):
    """
    Classifies a pair of embeddings via [e_v || e_c || |e_v-e_c| || e_v*e_c].

    This is the standard verification comparison signal, operating on
    embeddings rather than on raw per-timestep features, so that the
    encoder producing those embeddings is the only thing an ablation
    swaps out.
    """

    def __init__(self, embed_dim, hidden=64, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, e_v, e_c):
        combined = torch.cat([e_v, e_c, torch.abs(e_v - e_c), e_v * e_c], dim=1)
        return self.mlp(combined)


def _init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.LSTM):
            continue  # PyTorch default LSTM init, matches full_pipeline.py
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class CNNOnlyModel(nn.Module):
    """Variant 1: CNN spatial encoder only -- no temporal modeling."""

    def __init__(
        self,
        input_dim=78,
        hidden_dims=(64, 128),
        output_dim=128,
        verification_hidden=64,
        dropout=0.1,
    ):
        super().__init__()

        from models.gait_encoder import GaitEncoder

        self.encoder = GaitEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
        )
        self.head = EmbeddingComparisonHead(output_dim, verification_hidden, dropout)
        _init_weights(self)

    def _embed(self, x):
        # GaitEncoder returns (B, T, output_dim); pool over time to a vector.
        return self.encoder(x).mean(dim=1)

    def forward(self, video_features, claimed_features=None, mode="verification"):
        if mode != "verification" or claimed_features is None:
            raise ValueError("CNN-only requires verification mode")

        e_v = self._embed(video_features)
        e_c = self._embed(claimed_features)
        logits = self.head(e_v, e_c)
        probs = F.softmax(logits, dim=1)
        return {
            "verification": {
                "logits": logits,
                "probs": probs,
                "prediction": probs.argmax(dim=1),
            },
            "is_authentic": probs.argmax(dim=1),
            "similarity": probs[:, 1],
            "confidence": probs.max(dim=1).values,
            "video_embedding": e_v,
        }


class LSTMOnlyModel(nn.Module):
    """Variant 2: BiLSTM only -- local temporal patterns, no CNN/Transformer."""

    def __init__(
        self,
        input_dim=78,
        lstm_hidden=64,
        lstm_layers=1,
        verification_hidden=64,
        dropout=0.1,
    ):
        super().__init__()

        from models.temporal_model import BiLSTMEncoder

        self.bilstm = BiLSTMEncoder(
            input_dim=input_dim,
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout,
        )
        embed_dim = self.bilstm.output_dim  # hidden_dim * num_directions
        self.head = EmbeddingComparisonHead(embed_dim, verification_hidden, dropout)
        _init_weights(self)

    def _embed(self, x):
        _, hidden = self.bilstm(x)
        return hidden

    def forward(self, video_features, claimed_features=None, mode="verification"):
        if mode != "verification" or claimed_features is None:
            raise ValueError("LSTM-only requires verification mode")

        e_v = self._embed(video_features)
        e_c = self._embed(claimed_features)
        logits = self.head(e_v, e_c)
        probs = F.softmax(logits, dim=1)
        return {
            "verification": {
                "logits": logits,
                "probs": probs,
                "prediction": probs.argmax(dim=1),
            },
            "is_authentic": probs.argmax(dim=1),
            "similarity": probs[:, 1],
            "confidence": probs.max(dim=1).values,
            "video_embedding": e_v,
        }


class TransformerOnlyModel(nn.Module):
    """Variant 3: Transformer only -- global attention, no CNN/LSTM."""

    def __init__(
        self,
        input_dim=78,
        d_model=128,
        nhead=4,
        num_layers=2,
        verification_hidden=64,
        dropout=0.1,
    ):
        super().__init__()

        from models.temporal_model import TransformerEncoder

        self.transformer = TransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.head = EmbeddingComparisonHead(d_model, verification_hidden, dropout)
        _init_weights(self)

    def _embed(self, x):
        _, emb = self.transformer(x)
        return emb

    def forward(self, video_features, claimed_features=None, mode="verification"):
        if mode != "verification" or claimed_features is None:
            raise ValueError("Transformer-only requires verification mode")

        e_v = self._embed(video_features)
        e_c = self._embed(claimed_features)
        logits = self.head(e_v, e_c)
        probs = F.softmax(logits, dim=1)
        return {
            "verification": {
                "logits": logits,
                "probs": probs,
                "prediction": probs.argmax(dim=1),
            },
            "is_authentic": probs.argmax(dim=1),
            "similarity": probs[:, 1],
            "confidence": probs.max(dim=1).values,
            "video_embedding": e_v,
        }


class FullHybridModel(nn.Module):
    """
    Variant 4: CNN encoder -> (BiLSTM + Transformer) dual-path -> fused
    embedding, using the SAME EmbeddingComparisonHead as the other three
    variants, so all four are compared on equal footing.

    Note this is deliberately NOT models.full_pipeline.GaitDeepfakeDetector.
    That production model's verification-mode logits come from a separate
    diff_conv/diff_classifier head applied to raw features and never consume
    this branch's embedding (see NOTES.md Section 2.1) -- reusing it here
    would just reproduce the original bug for the fourth variant. This class
    reuses the same GaitEncoder / DualPathTemporalModel building blocks, wired
    the same way every other variant in this ablation is wired.
    """

    def __init__(
        self,
        input_dim=78,
        encoder_hidden_dims=(64, 128),
        encoder_output_dim=128,
        lstm_hidden=64,
        lstm_layers=1,
        transformer_d_model=128,
        transformer_heads=4,
        transformer_layers=2,
        embedding_dim=128,
        verification_hidden=64,
        dropout=0.1,
    ):
        super().__init__()

        from models.gait_encoder import GaitEncoder
        from models.temporal_model import DualPathTemporalModel

        self.encoder = GaitEncoder(
            input_dim=input_dim,
            hidden_dims=encoder_hidden_dims,
            output_dim=encoder_output_dim,
            dropout=dropout,
        )
        self.temporal = DualPathTemporalModel(
            input_dim=encoder_output_dim,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            transformer_d_model=transformer_d_model,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            output_dim=embedding_dim,
            dropout=dropout,
        )
        self.head = EmbeddingComparisonHead(embedding_dim, verification_hidden, dropout)
        _init_weights(self)

    def _embed(self, x):
        encoded = self.encoder(x)
        _, embedding = self.temporal(encoded)
        return embedding

    def forward(self, video_features, claimed_features=None, mode="verification"):
        if mode != "verification" or claimed_features is None:
            raise ValueError("Full Hybrid requires verification mode")

        e_v = self._embed(video_features)
        e_c = self._embed(claimed_features)
        logits = self.head(e_v, e_c)
        probs = F.softmax(logits, dim=1)
        return {
            "verification": {
                "logits": logits,
                "probs": probs,
                "prediction": probs.argmax(dim=1),
            },
            "is_authentic": probs.argmax(dim=1),
            "similarity": probs[:, 1],
            "confidence": probs.max(dim=1).values,
            "video_embedding": e_v,
        }


# ============================================================
# Training & Evaluation Helpers
# ============================================================


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train model for one epoch."""
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch in loader:
        video = batch["video_features"].to(device)
        claimed = batch["claimed_features"].to(device)
        labels = batch["label"].to(device).squeeze()

        optimizer.zero_grad()
        output = model(video, claimed, mode="verification")
        logits = output["verification"]["logits"]
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * video.size(0)
        preds = output["is_authentic"]
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total * 100


def evaluate_model(model, loader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels, all_scores = [], [], []

    with torch.no_grad():
        for batch in loader:
            video = batch["video_features"].to(device)
            claimed = batch["claimed_features"].to(device)
            labels = batch["label"].to(device).squeeze()

            output = model(video, claimed, mode="verification")
            logits = output["verification"]["logits"]
            loss = criterion(logits, labels)

            total_loss += loss.item() * video.size(0)
            preds = output["is_authentic"]
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(output["similarity"].cpu().numpy())

    # Compute additional metrics
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    accuracy = correct / total * 100
    f1 = f1_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, zero_division=0) * 100

    try:
        auc = roc_auc_score(all_labels, all_scores) * 100
    except:
        auc = 0.0

    return {
        "loss": total_loss / total,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc,
    }


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# Main Ablation
# ============================================================


def run_ablation(args):
    """Run complete ablation study."""

    import warnings

    warnings.filterwarnings("ignore")

    print("\n" + "=" * 70)
    print("  ABLATION STUDY -- Gait-Based Deepfake Detection")
    print("=" * 70)
    print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Epochs per variant: {args.epochs}")
    print("=" * 70)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
        torch.backends.cudnn.benchmark = True

    # Data
    print(f"\n  Loading data from {args.features_file}...")
    train_loader, val_loader, train_persons, val_persons = create_data_loaders(
        features_file=args.features_file,
        enrolled_identities_file=args.enrolled_file,
        batch_size=args.batch_size,
        train_ratio=0.8,
        mode="verification",
        num_workers=0,
        seed=42,
    )
    print(f"  Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")
    print(f"  Train persons: {train_persons}")
    print(f"  Val persons: {val_persons}")

    # Define variants
    variants = OrderedDict(
        {
            "CNN-Only": lambda: CNNOnlyModel(
                input_dim=78,
                hidden_dims=(64, 128),
                output_dim=128,
                verification_hidden=64,
                dropout=args.dropout,
            ),
            "LSTM-Only": lambda: LSTMOnlyModel(
                input_dim=78,
                lstm_hidden=64,
                lstm_layers=1,
                verification_hidden=64,
                dropout=args.dropout,
            ),
            "Transformer-Only": lambda: TransformerOnlyModel(
                input_dim=78,
                d_model=128,
                nhead=4,
                num_layers=2,
                verification_hidden=64,
                dropout=args.dropout,
            ),
            "Full Hybrid": lambda: FullHybridModel(
                input_dim=78,
                encoder_hidden_dims=(64, 128),
                encoder_output_dim=128,
                lstm_hidden=64,
                lstm_layers=1,
                transformer_d_model=128,
                transformer_heads=4,
                transformer_layers=2,
                embedding_dim=128,
                verification_hidden=64,
                dropout=args.dropout,
            ),
        }
    )

    # Results storage
    results = {}
    output_dir = Path(args.output_dir) / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train each variant
    for variant_idx, (name, model_fn) in enumerate(variants.items(), 1):
        print(f"\n{'='*70}")
        print(f"  VARIANT {variant_idx}/4: {name}")
        print(f"{'='*70}")

        torch.manual_seed(42)
        np.random.seed(42)

        model = model_fn().to(device)
        n_params = count_parameters(model)
        print(f"  Parameters: {n_params:,}")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=7, min_lr=args.lr * 0.01
        )

        best_val_acc = 0.0
        best_metrics = {}
        patience_counter = 0

        for epoch in range(args.epochs):
            t0 = time.time()

            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_metrics = evaluate_model(model, val_loader, criterion, device)

            scheduler.step(val_metrics["accuracy"])
            elapsed = time.time() - t0

            # Progress update every 5 epochs or at end
            if (epoch + 1) % 5 == 0 or epoch == 0 or (epoch + 1) == args.epochs:
                print(
                    f"  Epoch {epoch+1:3d}/{args.epochs} | "
                    f"Train: {train_acc:.1f}% | "
                    f"Val: {val_metrics['accuracy']:.1f}% (F1={val_metrics['f1']:.1f}%, AUC={val_metrics['auc']:.1f}%) | "
                    f"{elapsed:.1f}s"
                )

            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                best_metrics = val_metrics.copy()
                best_metrics["train_accuracy"] = train_acc
                best_metrics["params"] = n_params
                patience_counter = 0

                # Save best checkpoint
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "metrics": best_metrics,
                        "variant": name,
                    },
                    str(output_dir / f'{name.lower().replace(" ", "_")}_best.pth'),
                )
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        results[name] = best_metrics
        print(
            f"\n  [ok] {name} Best: Acc={best_val_acc:.2f}%, F1={best_metrics['f1']:.2f}%, AUC={best_metrics['auc']:.2f}%"
        )

    # ============================================================
    # Summary Table
    # ============================================================
    print(f"\n\n{'='*70}")
    print("  ABLATION STUDY RESULTS")
    print(f"{'='*70}")
    print(
        f"  {'Variant':<20s} {'Params':>10s} {'Accuracy':>10s} {'F1':>8s} {'AUC':>8s} {'Prec':>8s} {'Recall':>8s}"
    )
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for name, m in results.items():
        print(
            f"  {name:<20s} {m['params']:>10,d} {m['accuracy']:>9.2f}% {m['f1']:>7.2f}% "
            f"{m['auc']:>7.2f}% {m['precision']:>7.2f}% {m['recall']:>7.2f}%"
        )

    # Highlight best
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    print(
        f"\n  * Best variant: {best_name} ({results[best_name]['accuracy']:.2f}% accuracy)"
    )

    # Component contribution analysis
    full = results.get("Full Hybrid", {})
    if full:
        print(f"\n  Component Contribution (vs Full Hybrid {full['accuracy']:.2f}%):")
        for name, m in results.items():
            if name != "Full Hybrid":
                delta = full["accuracy"] - m["accuracy"]
                sign = "+" if delta >= 0 else ""
                print(
                    f"    Removing {'CNN' if 'CNN' not in name else 'LSTM' if 'LSTM' not in name else 'Transformer'}: "
                    f"{name} -> {m['accuracy']:.2f}% (delta = {sign}{delta:.2f}%)"
                )

    print(f"\n{'='*70}")
    print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    # Save results
    results_file = str(output_dir / "ablation_results.json")
    serializable = {
        "_meta": {
            "schema": "embedding-comparison-v2",
            "note": (
                "Each variant encodes video and claimed sequences through "
                "its own branch and classifies a comparison of the two "
                "embeddings. Fixes the v1 script, where all four variants "
                "shared an identical raw-feature diff_conv/diff_classifier "
                "head and the named branch never reached the logits. "
                "See NOTES.md Section 2.2 / paper.tex Section V-D."
            ),
            "generated": datetime.now().isoformat(),
        }
    }
    for name, m in results.items():
        serializable[name] = {
            k: float(v) if isinstance(v, (np.floating, float)) else int(v)
            for k, v in m.items()
        }
    with open(results_file, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved to {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Ablation Study -- Gait Deepfake Detection"
    )
    parser.add_argument(
        "--features_file", type=str, default="data/gait_features/gait_features.pkl"
    )
    parser.add_argument(
        "--enrolled_file",
        type=str,
        default="data/gait_features/enrolled_identities.pkl",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="outputs")

    args = parser.parse_args()
    run_ablation(args)


if __name__ == "__main__":
    logger = setup_logging("ablation")
    try:
        main()
    finally:
        close_logging(logger)

"""
Regenerate every data-backed figure in the paper.

    python scripts/generate_figures/generate_all.py

Each figure reads its numbers from the evaluation artefacts under outputs/;
none are hard-coded. Figures 1-4 are conceptual diagrams produced outside this
pipeline (see diagrams/GEMINI_PROMPTS.md) and are not generated here.
"""

import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MODULES = [
    ("fig05_roc", "Fig. 5  ROC analysis"),
    ("fig06_confusion", "Fig. 6  Pooled confusion matrices"),
    ("fig07_score_distribution", "Fig. 7  Score distributions"),
    ("fig08_loocv_spread", "Fig. 8  LOOCV fold dispersion"),
    ("fig09_ablation", "Fig. 9  Architectural ablation"),
    ("fig10_joint_importance", "Fig. 10 Joint attribution"),
    ("fig11_feature_groups", "Fig. 11 Feature-group attribution"),
]


def main():
    import importlib

    failures = []
    for name, title in MODULES:
        print("\n" + title)
        print("-" * len(title))
        try:
            importlib.import_module(name).main()
        except Exception:
            traceback.print_exc()
            failures.append(name)

    print("\n" + "=" * 60)
    if failures:
        print("FAILED: " + ", ".join(failures))
        return 1
    print("All %d data-backed figures written to figures/" % len(MODULES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

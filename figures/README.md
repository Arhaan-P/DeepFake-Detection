# figures/

Figures referenced by `deepfake_paper.tex`.

## Conceptual diagrams (Gemini-generated)

- `fig1_pipeline.png` — end-to-end system pipeline
- `fig2_threat_model.png` — face-swap threat model
- `fig3_keypoints.png` — the 12 selected MediaPipe landmarks
- `fig4_architecture.png` — network architecture (verification path vs. auxiliary branch)

Regenerate with the prompts in [`GEMINI_PROMPTS.md`](GEMINI_PROMPTS.md), saving
output under the exact filenames above. `deepfake_paper.tex` wraps each of
these in `\IfFileExists`, so the document compiles with labelled placeholder
boxes if a file is ever missing — no edit to `deepfake_paper.tex` required.

## Evaluation figures — generated from real evaluation output

Regenerate all seven with:

```
python scripts/generate_figures/generate_all.py
```

| File | Script | Reads |
|---|---|---|
| `fig5_roc.png` | `fig05_roc.py` | `outputs/evaluation/loocv/loocv_results.json` |
| `fig6_confusion.png` | `fig06_confusion.py` | `outputs/evaluation/loocv/loocv_results.json` |
| `fig7_score_distribution.png` | `fig07_score_distribution.py` | `outputs/evaluation/loocv/loocv_results.json` |
| `fig8_loocv_spread.png` | `fig08_loocv_spread.py` | `outputs/evaluation/loocv/loocv_results.json` |
| `fig9_ablation.png` | `fig09_ablation.py` | `outputs/ablation/ablation_results.json` + live parameter counts |
| `fig10_joint_importance.png` | `fig10_joint_importance.py` | `outputs/gradcam/aggregate/gradcam_results.json`, `data/gait_features/enrolled_identities.pkl` |
| `fig11_feature_groups.png` | `fig11_feature_groups.py` | `outputs/gradcam/aggregate/gradcam_results.json` |

No metric is hard-coded in any of these scripts. They fail loudly if a results
artefact is missing rather than falling back to a literal.

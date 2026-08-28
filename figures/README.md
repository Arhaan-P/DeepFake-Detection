# figures/

Figures referenced by `paper.tex`.

## Present — generated from real evaluation output

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

## Missing — conceptual diagrams to generate

These four are not produced by any script. Generate them with the prompts in
[`../diagrams/GEMINI_PROMPTS.md`](../diagrams/GEMINI_PROMPTS.md) and save them
here under exactly these names:

- `fig1_pipeline.png` — end-to-end system pipeline
- `fig2_threat_model.png` — face-swap threat model
- `fig3_keypoints.png` — the 12 selected MediaPipe landmarks
- `fig4_architecture.png` — network architecture (verification path vs. auxiliary branch)

`paper.tex` wraps each of these in `\IfFileExists`, so the document compiles
today with labelled placeholder boxes and picks up the real images
automatically once they are dropped in — no edit to `paper.tex` required.

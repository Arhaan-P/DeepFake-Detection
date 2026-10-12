# Gait Verification V2 — Phase 1B pretrained model integration report

**Date:** 12 October 2026 (Asia/Kolkata). **Branch:** `feat/gait-verification-v2`.
**HEAD:** `02b0312` (Phase 0 CUDA commit). Phase 1A and Phase 1B work is **uncommitted**.
**Authors:** OpenAI Codex (interrupted by usage exhaustion), then Claude Code
(recovery, independent verification, GREW retrieval and completion).

**PHASE 1B ENGINEERING: COMPLETE.** All four pretrained candidates load strictly
and run frozen, repeatable inference on real CUDA. **The official GPGait++ GREW
download is complete and integrity-checked.** No model is scientifically,
biometrically or production validated, and no threshold or winner has been selected.

## 1. Executive summary

Codex had implemented most of Phase 1B before running out of usage: the safe
extraction route, the CASIA-B GPGait++ integration, the variant-aware CUDA worker
and 28 tests. Claude Code recovered that work from the working tree without
reverting anything and re-ran every claim independently. All of them reproduced.
It then completed the GPGait++ GREW retrieval that Codex could not finish, safely
extracted the checkpoint, and validated it on CUDA.

| Candidate | Training data | Strict load | Frozen CUDA | Dim | Repeat Δ | Peak alloc | Status |
|---|---|---|---|---:|---:|---:|---|
| GaitPT-GREW | GREW | yes | yes | 256 | 0.0 | 28.4 MiB | engineering VERIFIED |
| GPGait-GREW | GREW | yes (1,310) | yes | 4,864 | 0.0 | 784.3 MiB | engineering VERIFIED |
| GPGait++ CASIA-B | CASIA-B | yes (1,010) | yes | 4,864 | 0.0 | 37.8 MiB | engineering VERIFIED; not a GREW comparator |
| **GPGait++ GREW** | GREW | **yes (2,570)** | **yes** | **4,864** | **0.0** | **798.3 MiB** | **engineering VERIFIED (new)** |

Regression suite: **95 passed, 0 failed, 2 deselected** (91 reproduced plus 4 new).
No training, fine-tuning, calibration fitting, threshold installation,
identification, unrestricted deserialization, commit or push took place. Original
checkpoints and videos are unchanged. All four registry entries remain `BLOCKED`
for deployment.

## 2. Recovered Codex work and actual repository state

Codex was interrupted by usage exhaustion during the GPGait++ GREW download, not
by a failed implementation. Recovery used the working tree as primary evidence,
not the handoff text. Nothing was reverted, reset, cleaned or committed.

Codex's own pre-edit snapshot (`outputs/gait_verification/phase1b-evidence/initial-status.txt`)
shows Phase 1B started on top of the uncommitted Phase 1A tree. The Phase 1B
delta therefore cannot be separated by Git alone; it was identified by diffing
against that snapshot and reading every changed file.

| File | State | Phase 1B content (Codex) |
|---|---|---|
| `gait_verification/backends/checkpoint_extract.py` | new, untracked | Hash-pinned static model-prefix extraction; `weights_only=True` is the only deserializer |
| `gait_verification/backends/fastpose_contract.py` | new, untracked | Pinned revision check; official YAML `model_cfg` per variant with dataset assertion |
| `gait_verification/backends/fastpose_infer_worker.py` | modified | Variant-aware strict loader, in-worker checksum recheck, tensor-only envelope, frozen/repeat/state audit, CUDA memory |
| `gait_verification/backends/fastpose_worker.py` | modified | `--dataset CASIA-B` path omits `Affine` (official CASIA evaluator) |
| `gait_verification/backends/gpgait.py` | modified | `GPGaitPlusPlusEncoder` (GREW), `GPGaitPlusPlusCASIAEncoder`, per-variant model IDs/datasets, worker timeouts |
| `gait_verification/backends/gaitpt.py` | modified (1A) | CUDA default with explicit failure; `min_frames=31` |
| `scripts/gait_verification/extract_model_only.py` | new | CLI over the audited extractor |
| `scripts/gait_verification/validate_model_integration.py` | new | Real pretrained CUDA audit on a provenance-checked RTMPose cache |
| `tests/gait_verification/test_phase1b.py` | new | 28 contract/security tests (mocked weights) |
| `configs/gait_verification/models.json` | modified | `gpgaitpp_casia` and `gpgaitpp_grew` entries; baseline model IDs |

No file contained incomplete code: all modules compiled, Ruff passed and the
selected suite reproduced Codex's **91 passed, 2 deselected** before any edit.
No Phase 1B report existed. Codex's evidence directories
(`phase1b-baseline-audit`, `phase1b-hardened-baselines`, `phase1b-casia-validation`,
`phase1b-evidence`) were preserved unchanged.

## 3. Phase 1A handoff status

Phase 1A remains as reported in the
[Phase 1A report](PHASE_1A_CAPTURE_PILOT_REPORT.md): engineering complete for
available fixtures; the representative four-view moving-camera pilot is
**pending** because no such recordings exist. The capture SOP stays a draft.
Phase 1A tests are part of the regression suite below and pass.

## 4. GaitPT-GREW verification

Independently re-executed (`outputs/gait_verification/phase1b-claude-verification/models.json`)
on the real 140-frame RTMPose cache of `data/videos/Aarav_S1.mp4`
(source SHA `3980dd51…`, cache SHA `4768d9d1…`, PTS and frame indices asserted
equal to a fresh `ffprobe`):

- Official revision `4977618a…`, checkpoint SHA `e6b287e2…`, strict load.
- 4,484,384 parameters; CUDA; `eval`; all `requires_grad=False`; state digest unchanged.
- 256-D, norm 1.0, repeat max delta 0.0; windowed aggregation repeat delta 0.0.
- 31-frame minimum accepted; short/missing/cropped/track-break inputs rejected on
  native and windowed paths; wrong checksum rejected; CUDA-unavailable raises.
- Peak allocated CUDA memory 28.4 MiB.
- Model ID identical to the registry:
  `gaitpt:e6b287e2…:grew-two-center30-height-v1:dd0b3c1a24b4`.

## 5. Vanilla GPGait-GREW verification

Same command and cache:

- Revision `d3ce6319…`; original SHA `929fc52b…`; model-only SHA `12c15082…`.
- 105,222,370 parameters; strict tensor-only load in the isolated worker; frozen.
- Official GREW `Affine → RescaleCenter → HOD → SkeletonInput`; input `(1,6,140,17,1)`;
  native output `(1,256,19)` → 4,864-D, norm 0.9999999; repeat delta 0.0.
- Window aggregation, 31-frame minimum, ten rejection checks (including explicit
  `device="cpu"` refusal) passed.
- Peak allocated CUDA memory 784.3 MiB (matches Codex's ~784 MiB).

## 6. GPGait++ CASIA-B checkpoint and CUDA inference

- Original `gpgaitpp-casia.pt`: 63,712,570 bytes, SHA `87eb74fa…`, unchanged
  (hash re-measured). Restricted loading of the original still fails on
  `GLOBAL __builtin__ getattr`; this was **not** bypassed.
- Static audit (§8) re-derived `model_end=173246` and pickle SHA `677def14…`
  exactly as pinned by Codex; the model section contains only tensor globals.
- **Re-extraction is deterministic:** a fresh extraction to a scratch path reproduced
  model-only SHA `0b13777d…` (21,307,301 bytes, 1,010 entries).
- Official CASIA-B config `configs/gpgait++/gpgait++.yaml`: `in_channels
  [2,64,64,128,256]`, `num_class 74`, `mask true`. A fresh architecture build
  has exactly 1,010 state entries and 5,227,539 parameters; strict load succeeds.
- Official CASIA-B evaluator preprocessing `RescaleCenter → HOD_MultiInput(coco) →
  SkeletonInput` — no `Affine` — matches the worker's `CASIA-B` path; defaults
  (`center='neck'`, `scale=225`) equal the official ones.
- Real CUDA run: input `(1,6,140,17,1)`, output `(1,256,19)` → 4,864-D, norm 1.0,
  repeat delta 0.0, state SHA `4154882a…` before = after, peak allocated 37.8 MiB.
- 31-frame, windowed, ten rejection checks: passed.

**Status:** engineering inference VERIFIED. It is a CASIA-B-trained candidate:
not a GREW comparator, and excluded from any matched GREW-pretraining comparison.

## 7. GPGait++ GREW download and checkpoint status

**Source.** The FastPoseGait README links the GPGait++ models on Google Drive and
on Baidu Netdisk. Baidu requires an account and client, so Google Drive was the
only usable official host. File ID `1PV3MnXIyrehx8TFHFxo7_yY4y1ItT7-Y`; the
server-reported filename is `GPGait++_GREW.pt`, `Last-Modified 10 Dec 2025`.
The server honours `Range` with an exact `Content-Range …/1308128306`.

**Codex's partial file.** `gpgaitpp-grew.incomplete-sparse` (219,152,384 logical
bytes) was inspected with `SEEK_DATA`/`SEEK_HOLE`. After the sequential
88,080,384-byte prefix, all 16 parallel 8 MiB ranges were only partially written
(3–5 MiB each, leaving holes). Its size is therefore not a valid resume offset,
and only the prefix was a candidate for reuse. The file is preserved unchanged
and was never loaded.

**Retrieval.** Claude Code's chunked downloader
(`outputs/gait_verification/phase1b-grew-download/download_gpgaitpp_grew.py`)
accepted 16 MiB ranges only after HTTP 206, an exact `Content-Range`, an exact
length and an fsync, journaling each chunk's SHA-256. It verified 25 chunks
(≈488 MB, including the reused prefix) at ≈0.9 MB/s, then hit read timeouts when
the owner switched networks. The owner then completed a **fresh full sequential
download** with `gdown` over TLS on the new network.

**Integrity evidence for the published file**
(`.cache/gait_verification/checkpoints/gpgaitpp-grew.pt`, now mode 0444):

- Size exactly **1,308,128,306** bytes, equal to the publisher listing and the server `Content-Range` total.
- SHA-256 **`eb37d274640d59665e500f8881811b3effd285302f5e0b9164ae3e22cb74e8d7`** (local provenance hash; the publisher supplies no digest).
- **All 8,041 ZIP member CRC-32 checks pass** (`zipfile.testzip`), covering every stored byte.
- The 25 independently verified chunks from the separate transfer are byte-identical to the corresponding ranges.
- Restricted `torch.load(weights_only=True)` of the original still fails on `GLOBAL getattr`, as expected.

**Static audit** (`static-audit.json`): pickle SHA `7b1f91ab…`, head
`PROTO 2, EMPTY_DICT, BINPUT 0, MARK, 'model'`, keys model@6, optimizer@423713,
scheduler@927602, iteration@1148863. Model-section globals are exactly
`OrderedDict, FloatStorage, LongStorage, _rebuild_tensor_v2`. `getattr`, `Adam`,
`OneCycleLR`, `defaultdict` and `_rebuild_parameter` occur only in the training tail.
After review, the artifact was pinned in `AUDITED` (`model_end=423713`, 2,570 entries).

**Extraction** (`extraction.json`): unchanged `weights_only=True`; 2,570 tensor-only
entries; model-only file `gpgaitpp-grew-model-only.pt`, 436,058,125 bytes, SHA
**`be1bcbf90dc8635e72d8f3e99818a10edb8c37c7322f1a90bcebc7dc1930a45b`**. The
original's hash is unchanged after extraction.

**Architecture and preprocessing.** Official `configs/gpgait++/gpgait++_GREW.yaml`
(`mask: false`, 20,000 classes). A fresh build has exactly 2,570 entries and
108,777,035 parameters, and strict loading succeeds. The GREW evaluator transform
`Affine → RescaleCenter → HOD_MultiInput → SkeletonInput` matches vanilla GPGait GREW
and the worker's `GREW` path.

**Real CUDA inference** (`outputs/gait_verification/phase1b-grew-validation/models.json`,
same RTMPose cache as §4):

- Input `(1,6,140,17,1)`, output `(1,256,19)`, giving a 4,864-D embedding with norm 1.0.
- Repeat delta 0.0 within the worker and across separate worker processes.
- Model state SHA unchanged; `training=False`; all parameters frozen.
- Peak allocated CUDA memory 798.3 MiB (832.0 MiB reserved); two forwards took 0.363 s.
- The 31-frame minimum is accepted; windowed aggregation is repeatable (delta 0.0).
- Short, missing-joint, cropped and track-break inputs are rejected on native and
  windowed paths; a wrong checksum and an explicit `device="cpu"` are refused.
- Model ID `gpgaitpp:be1bcbf9…:grew-affine-rescalecenter-hod-skeleton-v1:422313e7684b`.

**Status:** engineering inference VERIFIED. Biometric, domain and deployment
validation remain pending.

## 8. Safe deserialization methodology

1. **Never** `weights_only=False`, no custom unpickler, no `add_safe_globals`.
2. Static opcode walk (`pickletools.genops`, which does not execute opcodes) via
   `static_pickle_audit` (`scripts/gait_verification/audit_checkpoints.py
   --static-pickle`). It records the pickle SHA, the head
   `PROTO 2, EMPTY_DICT, BINPUT 0, MARK, 'model'`, the top-level key positions and
   the globals in the model section versus the whole pickle.
3. A human-reviewed `AUDITED` entry pins original SHA/bytes, pickle SHA,
   `model_end` and the expected state-entry count. Unaudited files are refused
   before any load.
4. The prefix up to the `'optimizer'` key is closed with `SETITEMS, STOP`; its
   globals must be a subset of `{OrderedDict, _rebuild_tensor_v2, FloatStorage,
   LongStorage}`; `pickletools.dis` checks stack/memo consistency.
5. PyTorch's unchanged `torch.load(weights_only=True)` deserializes the rebuilt
   archive. The envelope must be exactly `{"model": state}` with `str → Tensor`
   items and the pinned count. Tensors are cloned and re-saved tensor-only to a
   new file (`open(..., "xb")`); the original is never written.
6. Inference workers re-hash the checkpoint at load time, refuse non-model-only
   envelopes and load with `strict=True`.

Cross-recipe strict-load checks (actual checkpoints): GPGait++-GREW config with
CASIA weights, GPGait++-GREW config with vanilla GPGait weights, CASIA config with
vanilla weights, and vanilla config with CASIA weights all **fail** with missing or
unexpected keys and size mismatches. A checkpoint therefore cannot silently serve
as another candidate.

## 9. Official model configurations

| Candidate | Official config | `in_channels` | `num_class` | `mask` | State entries | Parameters |
|---|---|---|---:|---|---:|---:|
| GPGait GREW | `configs/gpgait/gpgait_GREW.yaml` | 2,64,128,128,128,128,256,256 | 20000 | n/a | 1,310 | 105,222,370 |
| GPGait++ GREW | `configs/gpgait++/gpgait++_GREW.yaml` | 2,64,128,128,128,128,256,256 | 20000 | false | 2,570 | 108,777,035 |
| GPGait++ CASIA-B | `configs/gpgait++/gpgait++.yaml` | 2,64,64,128,256 | 74 | true | 1,010 | 5,227,539 |
| GaitPT GREW | GaitPT `configs/gaitpt_config.yaml` | — | — | — | — | 4,484,384 |

All FastPoseGait configs are at revision `d3ce631971a01d806eeb3a154774d482c0951813`;
the worker refuses a different or dirty checkout. A new regression test asserts that
each encoder's model-ID fingerprint equals the pinned YAML `model_cfg`.

## 10. Embedding shapes and preprocessing

| Model ID prefix | Training data | Native input | Preprocessing recipe | Native output | Embedding |
|---|---|---|---|---|---:|
| `gaitpt` | GREW | `(2,30,18,2)` | `grew-two-center30-height-v1` | `(2,256)` | 256 |
| `gpgait` | GREW | `(1,6,T,17,1)` | `grew-affine-rescalecenter-hod-skeleton-v1` | `(1,256,19)` | 4,864 |
| `gpgaitpp` | GREW | `(1,6,T,17,1)` | `grew-affine-rescalecenter-hod-skeleton-v1` | `(1,256,19)` | 4,864 |
| `gpgaitpp_casia` | CASIA-B | `(1,6,T,17,1)` | `casia-rescalecenter-hod-skeleton-v1` | `(1,256,19)` | 4,864 |

Each embedding is flattened and L2-normalized. Model IDs bind variant, checkpoint
SHA, recipe, config fingerprint, pose recipe and variant/window recipe. The
enrollment store refuses mixed IDs. Equal 4,864-D shapes do **not** make
GPGait/GPGait++ embeddings interchangeable.

## 11. CUDA memory and runtime evidence

All runs used torch 2.10.0+cu128 on an NVIDIA GeForce RTX 3050 6GB Laptop GPU, with a
140-frame clip.

| Candidate | Parameters | Peak allocated | Notes |
|---|---:|---:|---|
| GaitPT-GREW | 4,484,384 | 28.4 MiB | in-process |
| GPGait-GREW | 105,222,370 | 784.3 MiB | isolated worker |
| GPGait++ CASIA-B | 5,227,539 | 37.8 MiB | isolated worker |
| GPGait++ GREW | 108,777,035 | 798.3 MiB (832.0 reserved) | isolated worker, 0.363 s for two forwards |

The large GREW GPGait-family counts come from the 20,000-class BNNeck classifier
that training kept; inference uses only the BNNeck features.

PyTorch peak allocated memory covers model weights plus activations on this
140-frame clip. Longer 30-second sequences are processed in windows and will grow
activation memory for the GPGait family. These numbers were measured on one
historical clip and are not throughput benchmarks.

## 12. Window-aggregation compatibility

Every validated candidate accepts the opt-in `WindowedEncoder`. Repeated windowed
encodings differ by 0.0, are unit-norm, and get a distinct `:windows=6409414b7493d6c2`
model ID, so windowed and native templates can never be pooled. `WindowedEncoder`
refuses an encoder whose minimum exceeds the window recipe's minimum.

## 13. Source modifications

**By Codex (verified, retained unchanged):** see §2.

**By Claude Code:**

- `gait_verification/backends/checkpoint_extract.py`: added `static_pickle_audit`
  (read-only opcode audit), and the reviewed GREW `AUDITED` entry.
- `scripts/gait_verification/audit_checkpoints.py`: additive `--static-pickle` flag.
- `tests/gait_verification/test_phase1b.py`: added a static-audit test (confirms no
  deserialization) and three tests that the model-ID fingerprint equals the pinned
  official config.
- `configs/gait_verification/models.json`: see §14.
- This report and dated amendments to the two project documents.

**Defects found:** no functional defect in Codex's working code. One verification
gap was closed: the hand-written model-ID fingerprint config was not tested against
the official YAML that the worker actually builds. It matched, and is now enforced
by test. Codex's GREW download method was unsafe to resume (16 parallel range
writers into one sparse file; every range only partially written). It was replaced,
not reused (§7).

## 14. Model registry updates

- `gpgaitpp_grew`: changed from UNAVAILABLE to engineering VERIFIED. The entry
  now records the original and model-only SHA, sizes, state entries, preprocessing,
  model IDs, CUDA device, 4,864-D output, `STRICT_VERIFIED` compatibility, download
  and safety evidence, and comparison eligibility as an independent GREW candidate.
- `gaitpt_grew`, `gpgait`, `gpgaitpp_casia`: added an `independent_verification`
  evidence path. Codex's values were reproduced exactly and left unchanged.
- Each entry keeps its own model ID, checkpoint fingerprint, training dataset,
  architecture, recipe, dimension, CUDA status, license and readiness.

All entries keep `readiness: BLOCKED` and `production_ready: false`, and every
`threshold` field is null. No winner is selected.

## 15. Tests performed and results

```bash
.venv-gait-v2-cuda/bin/python -m pytest -q tests/gait_verification tests/test_future_work.py \
  -k "not test_harness_end_to_end_on_synthetic_subjects and not test_metrics_against_known_values"
```

| Check | Result |
|---|---|
| Codex state, before Claude Code edits | 91 passed, 0 failed, 2 deselected |
| Final selected regression suite | **95 passed, 0 failed, 2 deselected**, 1 upstream Transformer warning |
| `test_phase1b.py` alone | 32 passed (28 Codex + 4 new) |
| Ruff check / format (changed files) | passed |
| `compileall` | passed |
| `git diff --check`, plus whitespace scan of untracked files | passed |
| E0 paths (`models`, `utils`, training/inference, outputs, legacy tests) | no diff |
| Real CUDA audits | 4/4 candidates passed (§§4–7) |
| CASIA re-extraction determinism | identical SHA |
| Cross-recipe strict-load refusal | 4/4 refused |

Not executed, by instruction: `test_harness_end_to_end_on_synthetic_subjects`
(neural training) and `test_metrics_against_known_values` (Platt fitting).

**Mocked/unit evidence** (pytest) is separate from **real CUDA evidence**
(`validate_model_integration.py` with actual pretrained checkpoints, RTMPose cache
and GPU), which is reported in §§4–7 and §11.

## 16. External blockers

- **None for Phase 1B engineering.** The GREW download blocker is resolved.
- Source-dataset reproduction: GREW/CASIA-B evaluation data are access-restricted.
- Four-view moving-camera capture pilot and the 40-participant cohort: not yet recorded.
- License clearance for any deployment or redistribution.
- `graphify` CLI absent, so the project graph could not be queried or updated.

## 17. License restrictions

- GaitPT: CC BY-NC-ND 4.0 (noncommercial, no derivatives).
- FastPoseGait (GPGait, GPGait++): the README restricts use to academic/noncommercial
  purposes; there is no standalone LICENSE file.
- Checkpoint redistribution rights are not granted by either source. Checkpoints stay
  in ignored `.cache/` and must not be committed or shared. Deployment needs
  separate license clearance.

## 18. Research limitations

- All inference evidence comes from **one historical 140-frame clip**. Finite,
  deterministic, normalized embeddings do **not** show recognition accuracy,
  domain fit for Android RTMPose poses, or comparative merit.
- Source-dataset reproduction (GREW/CASIA-B evaluation samples) is access-restricted
  and not performed. Native preprocessing parity is not semantic domain equivalence.
- CASIA-B (indoor, 124 subjects, 11 fixed views) and GREW (in-the-wild, ~26k
  identities) differ in data and preprocessing. No CASIA-vs-GREW claim is valid
  without a separately prespecified protocol.
- No new cohort, capture pilot, calibrated threshold, FAR/FRR, EER or AUC exists.
- Locally measured SHA-256 values are provenance records, not publisher signatures.

## 19. Phase 1C engineering readiness

**Phase 1B engineering is complete.** Four independent frozen candidates are
integrated with strict, safe, hash-pinned loading and a common embedding interface,
and each passes real CUDA checks: GaitPT-GREW, GPGait-GREW, GPGait++ GREW, and
GPGait++ CASIA-B (an ablation only).

**Phase 1C can begin** as engineering work, to the extent it does not need the new
recordings. A matched GREW-pretrained comparison set (GaitPT, GPGait, GPGait++) is
now available for the shared trial protocol. Anything that measures accuracy,
calibrates a threshold or picks a model still needs the capture pilot and
development data.

**Still pending for the real capture pilot:** 2–3 consenting volunteers, four
independent ~30 s views each with measured geometry and session metadata. Process
them through the frozen pose/quality config and visually adjudicate tracks, feet
and borders. Revise the provisional quality rules and SOP, then freeze the SOP
before mass collection. Pilot participants stay in development.

## 20. Exact next recommended action

Record the four-view pilot (2–3 volunteers, left45/right45/toward/away, about
30 s each, real metadata) following [CAPTURE_SOP_DRAFT.md](CAPTURE_SOP_DRAFT.md).
Run `compare_capture_pilot.py` on it, then run all three GREW-pretrained encoders
on the accepted windows with `validate_model_integration.py`. Check coverage, window
acceptance and CUDA memory on true 30-second inputs. Do not select a model or
threshold until development-split verification trials exist.

Housekeeping, at the owner's discretion: the 488 MB verified chunk cache
`.cache/gait_verification/checkpoints/gpgaitpp-grew-download/` and the quarantined
`gpgaitpp-grew.incomplete-sparse` are no longer needed for integrity. Both are
preserved and were not deleted.

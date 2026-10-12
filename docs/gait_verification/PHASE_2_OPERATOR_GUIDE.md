# Phase 2 operator and developer guide — dataset processing

12 October 2026. Every command below exists and is covered by a test (`tests/gait_verification/test_phase2.py`)
or was run in the real-CUDA CLI smoke described in the Phase 2 report. Nothing here needs a locked split,
a threshold or any owner decision. **Engineering tooling only: it does not validate a recording protocol or
measure recognition accuracy.**

All commands run from the repository root with `.venv-gait-v2-cuda/bin/python`. Abbreviation used below:

```bash
CL=".venv-gait-v2-cuda/bin/python scripts/gait_verification/evaluate_claims.py"
R=path/to/registry.json     # capture-registry-v1; raw rows are capture-manifest-v1 rows + 4 registry keys
D=path/to/data-root         # originals live here (read-only use); derivatives go to $D/derived/
W=path/to/work              # private derived artifacts (pose, quality, embeddings, ledger, run reports)
```

Keep `$D` and `$W` outside Git or under the ignored `data/gait_verification/` and `outputs/gait_verification/`.

## Rules the tools enforce

- **Explicit scope.** `plan`/`process`/`report`/`derive` act only on asset IDs given with `--asset` or
  `--assets-file` (one ID per line, `#` comments allowed). There is no "all" and no directory scan.
- **Registry is the source of truth.** Subject, session, date, view, consent reference and lineage come from the
  registry only. Nothing is read from filenames. A consent reference is not proof of consent.
- **Originals are read-only.** Selected sources and ancestor SHA-256s are verified before use and again after a run; derivative transformation manifests are checked before use. A work directory containing registered sources is refused.
- **No decision is defaulted.** Split counts/seed, impostor count and seed, and target FAR are unset; an
  operation that needs one stops and names its register ID (`OWN-01`…`OWN-05`).

## 1. Validate a capture registry

```bash
$CL validate --registry $R --data-root $D --measure --out reports/validate.json
```

Exit 0 = registry and files are consistent; 2 = problems (listed in the report); 3 = bad invocation.
`--measure` adds ffprobe facts. The report has one row per recording with four separate layers:
`media_validation` (bytes, size, orientation, frame timing, declared-vs-measured differences),
`operator_metadata` (missing optional fields are listed, never guessed), `pose_engineering_gate`
(`NOT_RUN` here) and `scientific_capture_suitability` (always `NOT_ASSESSED`).

If the registry itself fails to load (duplicate bytes, duplicate takes, conflicting subjects, placeholder
session or consent reference), the command exits 2 and the output lists which asset IDs collide.
Without `--data-root` only the registry structure is checked.

## 2. Prepare a processing plan and dry run

```bash
$CL plan --registry $R --data-root $D --work-dir $W --assets-file assets.txt \
         --model gaitpt --measure --out reports/plan.json
```

Planning reads source/cache state: it creates no processing directory, loads no model, enrolls nobody and installs nothing. An explicit `--out` publishes a new plan file. It
prints counts (assets, originals, derivatives, subjects, sessions), task states, CUDA jobs, blocked inputs, free
disk space, an optional storage estimate (`--measure`) and warnings. Task states:

| State | Meaning |
|---|---|
| `VALID` | sealed artifact exists and matches input, recipe and checksums: will be reused |
| `MISSING` | will be computed |
| `INVALID` | exists but fails a check (reason given): will be renamed aside and recomputed |
| `PENDING_UPSTREAM` | depends on an artifact that is missing or invalid |
| `PRIOR_PERMANENT_FAILURE` | failed permanently before on the same input (see §5) |
| `BLOCKED` | source missing, checksum mismatch, path rejected, or locked-final access denied |

`--model` may be repeated (`gaitpt`, `gpgait`, `gpgaitpp`, `gpgaitpp_casia`). Without `--model` only pose and
quality are planned. `--stages pose quality embed` restricts stages. Exit 2 if any input is blocked.
Keep the printed `plan_id` to guard execution (§3).

Pass `--split split.json` to apply the locked-final guard: assets of locked-final subjects, and subjects not in
the split, are blocked. Without a split, a registry row declaring `split_role: locked_final` is blocked. There
is deliberately no option to authorize access.

## 3. Run selected processing

```bash
$CL process --registry $R --data-root $D --work-dir $W --assets-file assets.txt \
            --model gaitpt --expect-plan-id <plan_id from the dry run>
```

Stages run in order: pose (real RTMPose on CUDA) → quality (CPU, provisional Phase 1A gate) → embedding per
model. Models load lazily, once per stage, and are released afterwards; a fully cached rerun loads nothing and
needs no GPU. `--max-gpu-jobs N` sets the cooperative per-Unix-user GPU slot bound (default 1; processes must share the same bound and lock root). If CUDA is unavailable or the
slots are taken, GPU tasks end as retryable failures; **there is no CPU fallback**. Ctrl-C/SIGTERM finish the
current task and stop cleanly (status `INTERRUPTED`). `--expect-plan-id` refuses to run if the plan changed
since the dry run. Exit 0 only when status is `COMPLETE`.

## 4. Resume an interrupted or partial run

Rerun the same `process` command. Completed, validated artifacts are skipped (`SKIPPED_VALID`); failed or
unfinished tasks are retried; permanent failures are not retried unless you add `--retry-permanent`. No other
flag is needed and a rerun never rewrites a valid artifact.

## 5. Inspect failures

- Run report: `$W/runs/<run_id>.json` → `status`, `task_counts`, `failures` (task, asset, reason, attempt),
  `blocked_inputs`, `originals.changed_during_processing`, `runtime` (GPU slot, backend, model IDs, versions).
- Audit log (append-only): `$W/ledger.jsonl`, e.g. `grep -h '"state":"FAILED' $W/ledger.jsonl`.
- Reasons: `INVALID_MEDIA`, `SOURCE_MISSING`, `SOURCE_CHECKSUM_MISMATCH` (permanent: fix the input);
  `CUDA_UNAVAILABLE_NO_CPU_FALLBACK`, `GPU_BUSY`, `CUDA_OR_HOST_OUT_OF_MEMORY`, `PROCESSING_ERROR:<type>`,
  `CACHE_INTEGRITY:<reason>` and `MODEL_RESOURCE_MISSING` (retryable); `DETECTOR_DOES_NOT_MATCH_PLAN`, `ENCODER_DOES_NOT_MATCH_PLAN` (the
  loaded weights are not the ones the plan pinned).
- Artifacts renamed to `*.invalid-N` were untrusted and are kept for inspection. Delete them yourself if wanted.

## 6. Reuse validated pose and embedding caches

- Pose artifacts live in `$W/pose/` (NPZ + sidecar with SHA-256s, recipe and extractor settings). Give the same
  directory to enrollment so nothing is re-extracted (an altered entry is refused):
  `evaluate`-side enrollment: `enroll_subject.py batch … --pose-cache-dir $W/pose`.
- Embeddings live in `$W/embeddings/<model>-<recipe>/`, one directory per recipe (checkpoint, window policy, pose
  backend, extractor settings). The exact relative path is in the plan (`embedding_stores`) and run report. Pass
  that directory to scoring: `$CL run … --query-store $W/embeddings/<model>-<recipe>`.
  Changing the checkpoint, window policy or pose backend creates a **new** directory; old ones are kept.
- Phase 2 processing requires sealed query records. Scoring checks seals whenever present or whenever processing fingerprints are declared; seal failures return an explicit processing error. Genuine unsealed Phase 1D query records remain readable with no retroactive integrity guarantee. Checksums detect corruption; they are not signatures.
- Prefer `process --model` over the older `evaluate_claims.py embed`: the old command's store key does not
  include the window policy or pose backend.

## 7. Coverage and quality summary

```bash
$CL report --registry $R --data-root $D --work-dir $W --assets-file assets.txt \
           --model gaitpt --out reports/coverage.json
```

Reads processing artifacts without changing them; `--out` publishes a new report. Per recording: pose state, the provisional pose-gate outcome (`USABLE_BY_PROVISIONAL_GATE` or
`NOT_USABLE_BY_PROVISIONAL_GATE`) with reasons, frames, accepted/rejected windows and their reasons, visibility,
ankle validity, detection and ambiguity fractions, tracking breaks, and per-model embedding status. Aggregates:
outcome counts and reason counts. **An unusable recording is retained and counted; it is not an identity
mismatch and not a processing failure.** Decoder, CUDA and I/O problems are listed separately.

## 8. Optional stabilized derivatives (RAW is the control)

```bash
$CL derive --registry $R --data-root $D --assets-file originals.txt \
           --variant vidstab-s10 --variant vidstab-s15 \
           --out-registry reports/registry-with-derivatives.json --out reports/derive.json
```

Creates FFmpeg vid.stab derivatives of the listed **originals** under `$D/derived/<variant>/<asset>/` with the
transformation manifest, then writes a **new** registry file (the old one is never edited) that is validated
before it is written. Rerunning reuses a derivative only when its manifest, S10/S15 configuration and bytes verify. `--split split.json` applies the protected-role guard; declared locked-final originals are also refused without a split. Then validate the new
registry (§1) and process a derivative by listing its asset ID. Variants are never generated implicitly, S10/S15
are not declared better than RAW, and a stabilized query carries a `:variant=` recipe suffix so it can never
match RAW enrollment references (the RAW-versus-stabilized policy, `OWN-08`, is unset).

## What these tools do not do

They do not create a split, trials, thresholds or calibration, and they do not enroll anyone. Those later steps
(`split` → `lock` → `trials` → `enroll_subject.py batch` → `run` → `compare`) are Phase 1D commands and need the
owner decisions in `DEFERRED_EVIDENCE_AND_DECISIONS.md`.

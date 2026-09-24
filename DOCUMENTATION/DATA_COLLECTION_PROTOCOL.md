# Data Collection Protocol (GaitDeepfake expansion)

For collaborators recording new walking data. It turns Section 8 of the future-work plan into instructions for recording sessions. The current corpus has 13 subjects, one camera, two views, indoors. Augmentation adds sequences but not independent gaits, so the most valuable addition is **new people, recorded under varied and labelled conditions**.

## 1. Before recording

- **Consent.** Every participant signs the consent form approved by your institution. The form must allow recording, research use, and release through IEEE DataPort. Record the form's ID in the manifest (`consent_id`), not the person's name.
- **Subject IDs.** Use a site prefix plus a number, e.g. `VITC014` or `IITM003`. Never put real names in file names or manifests. The mapping from ID to person stays with your site's PI.
- **Unique IDs across sites.** Each site gets its own prefix, so IDs never collide.

## 2. What to record per participant

**Minimum (one session):**

| View code | Camera placement | Takes |
|---|---|---|
| `S` | Side view, camera perpendicular to the walking path | 3 |
| `F` | Frontal view, walking towards the camera | 3 |

Each take is a straight walk of **at least 6 m / 8 seconds** at the participant's natural pace. The pipeline measures rhythm from full gait cycles. Many current clips last only 2.5–5 s, which gives 2–4 strides, and the gait-cycle analysis shows that clip-level rhythm parameters are unreliable at that length (`outputs/future_work/rhythm/`). Longer is the single cheapest improvement.

**Recommended additions.** Record these as extra takes, labelled in the manifest:

| Factor | Values | Why (plan Section 8) |
|---|---|---|
| Second session | a different day, ≥ 3 days apart | Separates identity from day-specific clothing, shoes and mood; the only honest test of re-identification |
| Oblique view | `OL` / `OR`, 30–45° | Tests view dependence |
| Walking speed | slow / natural / fast | Tests rhythm variability; do not mix speeds within one take |
| Distance | near (3 m) / medium (6 m) / far (10 m) | Tests scale effects |
| Camera | a second phone model | Tests sensor and domain robustness |
| Lighting | indoor / outdoor / low light / backlit | Tests pose robustness |
| Clothing | normal / jacket / loose trousers or long skirt | Tests appearance independence |
| Surface | tile / concrete / grass / corridor | Tests environmental variation |

## 3. Camera settings

- Phone on a tripod at hip height (about 1 m), landscape or portrait, but the same orientation for the whole session.
- 1080p at 30 or 60 fps. The pipeline reads the true frame rate; do not re-time the video.
- The whole body, including the feet, is visible for the entire walk. Keep only one person in the frame.
- Keep the original files. Do not trim, filter, stabilise or re-encode. If a file must be shared compressed, record the codec in the manifest.

## 4. File naming

```
{SubjectID}_{View}{Take}.mp4        e.g.  VITC014_S1.mp4, VITC014_F3.mp4, VITC014_OL2.mp4
```

The view code is **letters only** and the take number is **digits only**. The loader in `utils/gait_descriptors.py` splits on that boundary. Put every other condition in the manifest, not in the file name.

## 5. Manifest (`recordings.csv`, one row per file)

| Column | Required | Allowed values / format |
|---|---|---|
| `file` | yes | file name as recorded |
| `subject_id` | yes | site-prefixed ID |
| `site` | yes | institution code |
| `session` | yes | 1, 2, ... |
| `view` | yes | `F`, `S`, `OL`, `OR` |
| `take` | yes | integer |
| `camera` | yes | free text, e.g. `pixel7`, `iphone13` |
| `fps` | yes | nominal capture rate |
| `distance` | yes | `near`, `medium`, `far` |
| `speed` | yes | `slow`, `natural`, `fast` |
| `lighting` | yes | `indoor`, `outdoor`, `lowlight`, `backlit` |
| `clothing` | yes | `normal`, `jacket`, `loose` |
| `surface` | yes | `tile`, `concrete`, `grass`, `corridor`, `other` |
| `codec` | no | e.g. `h264-original`, `h265`, `whatsapp` |
| `age_band` | no | `18-24`, `25-34`, `35-49`, `50+` |
| `sex` | no | `F`, `M`, `X`, or blank |
| `consent_id` | yes | consent form reference |
| `notes` | no | free text, e.g. "stumbled at 4 s" |

Validate before uploading:

```powershell
python scripts/future_work/validate_recordings.py --manifest recordings.csv --videos_dir path\to\videos
```

The validator checks the schema, naming, readability, duration and frame rate. It prints a subject × condition coverage table, so gaps are visible before anyone leaves the recording site.

## 6. Using the new data

1. Extract a full-frame-rate pose cache:
   `python scripts/future_work/extract_pose_cache.py --backend mediapipe_lite --videos_dir <dir>`
2. Any manifest column becomes a cross-condition protocol:
   `python scripts/future_work/run_experiment.py --name cross_light --set metadata_csv=recordings.csv protocol=cross:lighting:indoor:outdoor`
3. With two sessions per person, enrol on session 1 and verify session 2: `protocol=cross:session:1:2`. This is the most informative single experiment the expanded data allows.

## 7. Face-swap clips (for E9)

For every generator used, record the generator name and version, model, mask type, and output codec. Name each clip `{BodyID}_body_{FaceID}_face_{generator}.mp4`. Add it to a manifest with columns `video,body_identity,face_identity,generator,codec`, as consumed by `scripts/future_work/generator_robustness.py`. Only swap between consenting participants.

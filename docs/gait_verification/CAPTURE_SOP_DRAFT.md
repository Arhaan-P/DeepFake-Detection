# Capture SOP draft — four-view gait pilot

12 October 2026. **Draft for a 2–3 volunteer pilot, not a frozen collection protocol.**
The target study is about 40 participants, four independent original videos each,
about 30 seconds each. We first need to test the four views on the actual phone.
The current historical videos do not validate this recording procedure.

## Before recording

- Get consent for recording and gait biometric research. Use a code such as S001.
  Keep names and consent records in restricted storage outside the public repo.
  Agree access, retention and deletion arrangements with the supervisor.
- Record the actual date, time with time zone, session ID, operator code and phone
  settings. A session is a real recording occasion. Four files recorded together
  are four takes in one session, not four different days.
- Choose a long, flat, well-lit route. Remove obstacles and keep other people out
  of the background. Do not block a corridor or use stairs for moving-camera takes.
- Clean the lens. Use the same rear main 1× lens and phone mode for a matched pilot.
  Try 1080p60 if the phone supports it in that mode. If only 30 fps is available,
  record that fact. Inspect the actual file timestamps later; the camera menu is
  not proof of constant 60 fps.
- Keep orientation consistent within a take. Portrait is allowed. Do not switch
  lenses, zoom, resolution or stabilization modes during the take. Record the stock
  OIS/EIS setting and any automatic behavior; do not assume it can be disabled.
- Begin with the camera around the walker's waist/chest height (roughly 1–1.2 m)
  and 3–5 m away. These are trial starting values. Adjust before recording so the
  whole person, including both shoes, stays visible with generous space around
  the head and feet. Measure and write down the chosen height and distance.
- Ask the volunteer to walk naturally in their usual shoes. Do not ask them to
  exaggerate steps or time their gait to a beat. Note shoes, clothing and aids.

## Define the route and views consistently

Draw the walking direction as an arrow on a small route sketch. Left and right
mean **the walker's left and right while facing along that arrow**, not the
camera operator's left and right. The proposed oblique definition is **front-left
and front-right**: the line from the walker to the camera is 45° from the walker's
forward direction. Whether these are the owner's intended left/right views is
still an operator decision. Label this exact definition in the manifest.

Mark a central position and measure the camera's side and forward offsets. Equal
side and forward offsets make a 45° angle at that central position. A stationary
camera does not maintain 45° as the walker travels. Write down the angle range
or geometry, rather than labeling the entire changing view as exactly 45°.

For the pilot, compare these two possible oblique procedures on separate original
files. Do not treat either as the approved mass-collection method yet:

1. **Stationary oblique camera:** place the camera at equal side/forward offsets
   from the central route mark. Record distance, camera height and walking zone.
   Keep the lens fixed. The angle changes during the take; annotate the central
   zone and approximate angle range. Do not splice repeated passes together.
2. **Moving oblique camera:** use a parallel operator lane, keeping approximately
   equal side/forward offsets as the walker moves. Measure the intended offsets
   and check them at the start, middle and end. This is a candidate for maintaining
   the angle, but introduces operator motion and needs a safe, wide route.

| Planned view | Pilot instructions |
|---|---|
| left45 | Use the stated front-left definition and one labeled oblique procedure. Keep both feet visible; note any far-leg occlusion. |
| right45 | Mirror the geometry on the walker's right. Use the same distance, height and procedure as left45. |
| toward | Walker moves forward on the marked route. Camera stays ahead, facing the walker; operator moves backward only on a checked clear route with a spotter. Maintain full-body framing and roughly constant distance. Stop if unsafe; a stationary-camera alternative must be labeled as a separate condition. |
| away | Camera stays behind the walker, facing along the route. Operator follows forward at roughly constant distance. Keep both shoes visible and avoid sudden accelerations or turns. |

Film about 30 seconds of continuous natural walking per take. A route too short
for this needs a revised pilot setup, not an instruction to turn repeatedly inside
an otherwise unlabeled take. Note starts, stops, turns and occlusions. If a take
fails, keep it and record a new take with a new ID and reason.

## Quick review immediately after each take

- Is the full body visible, with both shoes clear through multiple walking cycles?
- Does the frame have enough spare room above the head, below the shoes and at the
  sides? Pilot a margin of roughly 10% of frame height below the shoes where practical.
- Is the walker large enough for reliable knees/ankles? Check the actual pose overlays.
- Are there bystanders, strong blur, abrupt reframing or changing lenses/exposure?
- Does the operator maintain the intended route and distance safely?
- Did the app save the original file, rather than just a cloud-edited copy?

## Preserve files and metadata

Copy the original phone file without editing it. Hash it with SHA-256. Keep the
original read-only in project storage. Give each original a unique take ID such
as `S001_session01_toward_take01`; use session02 only for an actual separate occasion.
Never overwrite an original with stabilization, an overlay or a recompressed copy.

Complete all fields in `configs/gait_verification/capture_schema.json`: participant,
session, actual date, view and its angle definition, walking direction, operator
motion, measured distance, phone/lens/settings/stabilization, resolution/FPS,
actual timestamps reference, duration, file path/hash, consent reference, operator
code, notes and review result. Add camera height, route sketch, start/middle/end
geometry and shoes to recording notes. Unknown measurements must be marked unknown,
not guessed. New study consent/date/session fields must actually be recorded.

Pilot volunteers and all their takes/derivatives stay in development if they
influence settings. Windows, stabilized copies and augmentations stay with the
same original and split. They are not extra independent recordings. If all four
views come from one day, later results can test different takes/views, not different-day
robustness. A second-day subset would be useful if separately planned.

## Process the small pilot before filming everybody

Use one fixed detector, pose checkpoint and quality configuration for RAW, S10 and
S15. Keep RAW intact. S10/S15 are analysis copies using FFmpeg vid.stab with
shakiness 5, accuracy 15, mincontrast 0.2, smoothing 10 or 15, no automatic zoom,
black borders and bicubic interpolation. The engineer saves commands, hashes,
transforms, timestamps and per-frame/window diagnostics. Inspect overlays at the
beginning, middle and end and around flagged events. Check toes/shoes visually:
COCO17 provides ankle points, not heel/toe points.

**Provisional review targets, to test rather than assume:**

- Healthy media, matching frame counts and source-relative timestamps in derivatives.
- One unambiguous walker, no uncertain person changes.
- At least 80% complete confident in-frame COCO skeletons in each accepted 2-second
  window, using native score cutoff 0.3 and the declared 2-pixel border test.
- At least one unchanged contiguous 31-frame segment per accepted window. Never
  fill missing joint motion to pass the gate.
- For a roughly 30-second pilot take, aim for at least ten accepted 2-second windows
  spread through the take. Review shorter tails separately. This capture target is
  not a validated minimum for biometric verification.
- Review blur, apparent skeleton steps, body-scale changes and black borders.
  Pixel-motion and dark-edge measurements are flags for review, not proof of their cause.

Reject or recapture for persistent missing feet, too few valid windows, ambiguous
people, corrupt timing or unsafe recording. Do not report a gait identity mismatch
for these failures. Do not select stabilization on looks alone. The two historical
clips measured so far do not justify selecting S10 or S15; RAW remains the control.

## Decisions still needed before a final SOP

The owner/operator team must settle the exact meaning of left45/right45; static
versus moving oblique capture; route length and angle tolerance; camera height,
distance and framing; phone/lens and actual FPS mode; stock stabilization settings;
one day versus multiple days; extra same-view independent takes; and whether the
provisional pose targets are achievable across all four views. Record the pilot
results and revise this draft before collecting the full cohort.

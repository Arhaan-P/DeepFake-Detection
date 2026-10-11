# Project instructions for Codex — gait verification

For any changes involving gait models, pose extraction, stabilization, enrollment, biometric inference, recording manifests, experimentation or evaluation:

1. **Read both documents in full before editing:**
   - `docs/gait_verification/GAIT_VERIFICATION_REARCHITECTURE_HANDOFF.md`
   - `docs/gait_verification/GAIT_VERIFICATION_TECHNICAL_PRD.md`
2. Briefly state in your first task response the current scope, exact phase, non-goals, any contradictions with repository code and planned tests. Do not claim to have read the documents unless you actually opened them.
3. The only identity task is **1:1 claimed-identity verification**. On usable enrolled inputs return TRUE/FALSE; on low quality / unenrolled claim / errors return explicit non-Boolean status. **Never identify or rank other people.** Never claim a gait mismatch proves a deepfake.
4. Preserve audited E0 and historical experiments. Implement frozen pretrained GaitPT first; test GPGait++/GPGait as a challenger, not an assumed winner. No local training or irreversible edits unless the assigned phase requires it.
5. Original RGB files are immutable. Stabilization is a raw-versus-derivative experiment and must preserve lineage and split roles. Avoid data leakage from source derivatives and repeated windows.
6. Use the technical PRD as the implementation contract and the handoff for reasoning and historical context. Report any conflicting newer explicit user instruction for document updates instead of silently overwriting either document.
7. After implementation, identify changed files, commands/tests run, test failures, license/checkpoint blockers and remaining unvalidated requirements.

These directions supplement existing repo instructions; if another root AGENTS.md exists, **merge this content** rather than replacing it.

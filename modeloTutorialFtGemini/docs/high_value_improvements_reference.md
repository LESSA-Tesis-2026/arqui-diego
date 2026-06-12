# High-Value Feature Improvements (Post-Temporal)

This document captures the next two high-value feature improvements for `modeloTutorialFtGemini`, after temporal deltas were added.

Current status:
- Temporal features: implemented (`USE_TEMPORAL_FEATURES` in `config.py`).
- Remaining high-value improvements:
  1. Scale normalization
  2. Confidence / presence masking

---

## 1) Why these two are high-value

Your current extraction is already robust to global translation (nose-anchor coordinates), but two major sources of noise remain:

1. **Scale variation**: signer distance to camera, body size differences, framing zoom.
2. **Detection reliability variation**: occasional missing hands/face or unstable landmark confidence.

Scale normalization addresses (1). Masking addresses (2).

Together, they reduce nuisance variability and improve class separability without changing the task definition.

---

## 2) Improvement A: Scale normalization

### 2.1 Problem it solves

Nose-anchoring removes global offset, but coordinate magnitudes still change with signer-camera distance.

Example:
- Same sign, same motion pattern, but one person is closer to camera.
- Relative coordinates become larger in magnitude.
- Model sees avoidable variance unrelated to class semantics.

### 2.2 Core idea

Normalize all `(x, y, z)` coordinates by a body scale reference per frame:

- Preferred scale: shoulder distance between pose landmarks (left shoulder 11, right shoulder 12).
- Fallback if shoulders are unavailable: distance between hips (23, 24) or last valid scale.

Formula:

- `scale_t = ||pose[11] - pose[12]||_2`
- `coord_norm = coord / max(scale_t, eps)`

Recommended epsilon: `eps = 1e-6`.

### 2.3 Why this is justified

- Improves invariance to camera zoom and signer body size.
- Preserves trajectory shape while reducing magnitude drift.
- Typically improves transfer across users and sessions.

### 2.4 Expected impact in this project

- More stable training curves when mixing users/distances.
- Better confusion reduction on signs with similar shape but different amplitude.
- No direct increase in feature dimension (same vector size).

### 2.5 Implementation options in this repo

Option A (recommended): normalize inside `extract_keypoints()` in `config.py`.
- Pro: one source of truth used by both extraction and real-time inference.
- Pro: no extra post-processing pass required.

Option B: normalize in `utils/temporal_features.py`.
- Pro: togglable pipeline stage.
- Con: requires ensuring exact same order in extraction and inference call sites.

### 2.6 Risks and mitigations

Risk: scale instability when pose is poorly detected.
- Mitigation: use fallbacks and temporal smoothing of scale (EMA).

Risk: division spikes when shoulders overlap/mis-detect.
- Mitigation: clamp with epsilon and reuse previous valid scale.

---

## 3) Improvement B: Confidence / presence masking

### 3.1 Problem it solves

MediaPipe can intermittently miss landmarks (especially hands) under occlusion, motion blur, or out-of-frame conditions.
Current fallback is often zeros. Without explicit mask signals, zeros are ambiguous:
- "real near-zero coordinate" vs
- "landmark missing"

### 3.2 Core idea

Add explicit reliability channels so the model knows when data is missing/low-confidence.

There are two levels:

1. **Global region masks** (low-cost, recommended first)
   - `pose_present, face_present, lh_present, rh_present` (0/1).

2. **Per-landmark masks** (higher granularity)
   - one 0/1 mask per landmark point.

### 3.3 Why this is justified

- Reduces false learning from placeholder zeros.
- Helps the network condition decisions on observation reliability.
- Improves robustness in real-time where hand visibility flickers.

### 3.4 Expected impact in this project

- Fewer unstable predictions when hands leave frame briefly.
- Improved continuity in real-time decoding.
- Better tolerance to recording-condition variation.

### 3.5 Dimension tradeoffs (for this specific feature schema)

Current position-only base:
- Pose (33*4), Face (16*3), LH (21*3), RH (21*3) = **306**.

Current with temporal deltas enabled:
- **612**.

If adding masks:

A) Global region masks (4 values):
- Base: `306 + 4 = 310`
- With temporal: `620`

B) Per-landmark masks (33 + 16 + 21 + 21 = 91 values):
- Base: `306 + 91 = 397`
- With temporal: `794`

Recommendation:
- Start with **global masks** for fast iteration.
- Move to per-landmark masks only if needed.

### 3.6 Implementation options in this repo

Option A (recommended): add masks in `extract_keypoints()` output.
- Pro: same logic for dataset generation and real-time inference.
- Con: requires updating `BASE_LENGTH_KEYPOINTS` and regenerating `.h5`.

Option B: add masks in a dedicated utility transform in `utils/`.
- Pro: toggleable and modular.
- Con: greater risk of train/infer mismatch if call order diverges.

### 3.7 Risks and mitigations

Risk: feature-size churn while experimenting.
- Mitigation: keep a single config toggle and strict shape checks (already present in training loader).

Risk: over-parameterization with per-point masks on small datasets.
- Mitigation: start with global masks.

---

## 4) Recommended rollout strategy (safe and measurable)

1. **Baseline snapshot**
   - Keep current temporal setup.
   - Save metrics and confusion matrix as baseline.

2. **Add scale normalization only**
   - Keep dimensions unchanged.
   - Rebuild `.h5`, retrain, compare.

3. **Add global region masks**
   - Update base dimension and regenerate `.h5`.
   - Retrain and compare against step 2.

4. **Optional: per-landmark masks**
   - Only if class confusion remains tied to landmark dropout.

This isolates improvements and prevents confounded conclusions.

---

## 5) Evaluation protocol for justification

For each experiment variant, track:

1. Macro F1 and per-class precision/recall.
2. Confusion matrix changes on commonly confused classes.
3. Real-time stability metrics:
   - word flicker rate,
   - false trigger rate during idle,
   - repeat suppression quality.
4. Cross-session robustness:
   - same signer different day/lighting,
   - if available, different signer.

Acceptance criteria suggestion:
- Keep or improve macro F1, and
- reduce at least one of:
  - top-3 confusion pairs,
  - real-time flicker rate,
  - idle false positives.

---

## 6) Practical recommendation for this project right now

Given your current stage and data volume:

1. Implement **scale normalization first** (high gain, minimal dimensional change).
2. Implement **global region masks second** (high gain, modest dimensional increase).
3. Defer per-landmark masks until you need extra robustness.

This gives strong robustness gains with manageable implementation risk.

---

## 7) Operational notes

- Any change to final feature vector size requires regenerating `.h5` and retraining.
- Keep toggles centralized in `config.py` and transforms in `utils/`.
- Maintain strict shape validation in training/inference to avoid silent mismatches.

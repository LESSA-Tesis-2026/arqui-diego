# modeloTutorialFtGemini

A full pipeline for dynamic sign-language recognition using MediaPipe landmarks + sequence modeling (Bidirectional LSTM), from data capture to real-time translation.

This folder contains **two data acquisition routes** that converge into the same training and inference flow:

1. **Video-first route**: record `.avi` videos -> extract landmarks -> train model.
2. **Direct-H5 route**: capture landmarks directly from webcam into `.h5` -> train model.

---

## 1) What this project does

This project recognizes signs from webcam video by:

1. Detecting human landmarks with MediaPipe Holistic.
2. Converting each frame into a numeric feature vector.
3. Building variable-length sequences per sample.
4. Padding sequences to a fixed length (`MAX_FRAMES`).
5. Training a neural sequence classifier.
6. Running real-time translation with smoothing/voting.

The final output is a rolling sentence of recognized words in real time.

---

## 2) High-level architecture

### Inputs
- Webcam frames (for capture and real-time inference)
- Optional prerecorded videos (`videos/<word>/sample_*.avi`)

### Core representation
- Per-frame feature vector length: `LENGTH_KEYPOINTS = 306`
- Composition:
  - Pose: `33 * 4 = 132` (x, y, z, visibility)
  - Face subset: `16 * 3 = 48` (selected points only)
  - Left hand: `21 * 3 = 63`
  - Right hand: `21 * 3 = 63`

### Why 306 features (instead of full 1662)
- Lower dimensionality, faster training/inference
- Focus on landmarks most relevant to sign articulation
- Uses face subset + hands + body pose for a practical speed/accuracy tradeoff

### Coordinate strategy
`extract_keypoints()` in `config.py` converts coordinates to **relative positions anchored to the nose** (pose landmark 0). This reduces dependence on absolute camera position and improves robustness to user movement in frame.

---

## 3) Folder contents and responsibilities

### `config.py`
Shared config and reusable helpers across scripts:
- Global paths (`DATA_PATH`, `VIDEOS_FOLDER`, `MODEL_PATH`, etc.)
- Vocabulary (`WORDS`)
- Sequence dimensions (`MAX_FRAMES`, `LENGTH_KEYPOINTS`)
- New capture timing controls:
  - `PRE_RECORD_COUNTDOWN_SECONDS`
  - `RECORD_DURATION_SECONDS`
  - `DISPLAY_TIMER_DECIMALS`
- MediaPipe utility functions:
  - `mediapipe_detection(image, model)`
  - `extract_keypoints(results)`
  - `create_folder_if_not_exists(path)`

### `1a_capturar_videos.py`
Captures raw videos per word into `videos/<word>/`.

Current capture state machine:
- `IDLE`: waits for key `r`
- `COUNTDOWN`: pre-start window (5s by default)
- `RECORDING`: records fixed duration and auto-stops

Key points:
- Raw frames are saved (without drawn overlays)
- `q` exits cleanly from any state
- For class `"nada"`, target samples are increased (`x1.75`)

### `1b_procesar_videos_h5.py`
Batch-processing script that:
- Reads all `.avi` files per word
- Runs MediaPipe on each frame
- Saves sequence arrays into `data_h5/<word>.h5`

Each dataset in an `.h5` file corresponds to one sample/video (`sample_<idx>`).

### `1_capturar_muestras.py`
Alternative to video-first route:
- Captures keypoint sequences directly to `.h5`
- No intermediate `.avi`

Useful if you do not need raw videos and want to build training data faster.

### `1c_unir_datasets.py`
Utility script to merge sample datasets from one `.h5` file into another while renaming sample keys to avoid collisions.

### `2_entrenamiento_gpu.py`
Training script:
- Loads raw sequences from `data_h5/*.h5`
- Splits train/validation
- Applies data augmentation (noise-based)
- Pads sequences to `MAX_FRAMES`
- Builds and trains a Bidirectional LSTM model
- Saves:
  - model (`MODEL_PATH`)
  - training curves (`metricas_modelo/training_history.png`)
  - confusion matrix (`metricas_modelo/confusion_matrix.png`)

### `3_traduccion_tiempo_real.py`
Real-time inference:
- Sliding window over recent frames (`WINDOW_SIZE`)
- Pads to model input length (`MAX_FRAMES`)
- Predicts continuously
- Uses voting/smoothing buffer to emit stable words
- Maintains a short sentence buffer for display

### `requirements.txt`
Pinned dependency versions for reproducibility.

---

## 4) How everything is connected (data flow)

### Route A: Video-first pipeline
1. `1a_capturar_videos.py`
   - output: `videos/<word>/sample_*.avi`
2. `1b_procesar_videos_h5.py`
   - input: recorded `.avi`
   - output: `data_h5/<word>.h5`
3. `2_entrenamiento_gpu.py`
   - input: all `.h5` files from `data_h5`
   - output: trained `.keras` model + metrics images
4. `3_traduccion_tiempo_real.py`
   - input: webcam stream + trained model
   - output: live recognized sentence

### Route B: Direct-H5 pipeline
1. `1_capturar_muestras.py`
   - output: `data_h5/<word>.h5`
2. `2_entrenamiento_gpu.py`
3. `3_traduccion_tiempo_real.py`

### Optional maintenance step
- `1c_unir_datasets.py` can be used anytime after data collection to combine datasets.

---

## 5) Setup and environment

## Prerequisites
- Python 3.10+ recommended
- Webcam
- OS libraries compatible with OpenCV + MediaPipe

## Install
Run from this folder (`modeloTutorialFtGemini`):

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install --upgrade pip
pip install -r requirements.txt
```

Important: scripts use `ROOT_PATH = os.getcwd()`, so run commands from this folder to keep paths correct.

## Alternative setup with `uv` (recommended for speed)
If you use `uv`, you can install and manage Python + virtual environments quickly.

From this folder root (`modeloTutorialFtGemini`):

```bash
uv python install 3.11
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

If `uv` prints a PATH warning such as:
`~/.local/bin is not on your PATH`, run:

```bash
uv python update-shell
```

Or add this to `~/.zshrc` manually:

```bash
export PATH="/Users/fernandofigueroa/.local/bin:$PATH"
source ~/.zshrc
```

To exit the virtual environment:

```bash
deactivate
```

---

## 6) How to run each phase

## Phase 0: Configure vocabulary and settings
Edit `config.py`:
- Update `WORDS` list to your target vocabulary
- Tune:
  - `MAX_FRAMES`
  - `PRE_RECORD_COUNTDOWN_SECONDS`
  - `RECORD_DURATION_SECONDS`
  - `MODEL_PATH` if needed

## Phase 1A (video-first): Capture videos
```bash
python 1a_capturar_videos.py
```

Default interaction:
- `r`: start countdown
- countdown ends -> recording starts automatically
- recording auto-stops after configured duration
- `q`: quit

Expected output:
- `videos/<word>/sample_0.avi`, `sample_1.avi`, ...

## Phase 1B (video-first): Extract keypoints from videos
```bash
python 1b_procesar_videos_h5.py
```

Expected output:
- `data_h5/<word>.h5` with one dataset per sample

## Phase 1C (alternative route): Direct keypoint capture
If you prefer skipping `.avi` recording:
```bash
python 1_capturar_muestras.py
```

Expected output:
- same `data_h5/<word>.h5` format used by training

## Optional Phase: Merge datasets
Adjust filenames inside `1c_unir_datasets.py` and run:
```bash
python 1c_unir_datasets.py
```

## Phase 2: Train model
```bash
python 2_entrenamiento_gpu.py
```

Expected artifacts:
- model: `models/modelo_señas_lstm.keras`
- metrics:
  - `metricas_modelo/training_history.png`
  - `metricas_modelo/confusion_matrix.png`

## Phase 3: Real-time translation
```bash
python 3_traduccion_tiempo_real.py
```

Behavior:
- Shows predicted probabilities per class
- Emits stable words into sentence buffer using voting logic
- Press `q` to quit

---

## 7) Detailed model and inference behavior

## Training (`2_entrenamiento_gpu.py`)
- Loads sequence data class-by-class from `.h5`
- Uses train/validation split with stratification
- Augments only training set (noise variants)
- Pads/truncates all sequences to `MAX_FRAMES`
- Model architecture:
  - `Masking`
  - `SpatialDropout1D`
  - `Bidirectional LSTM (64)` + dropout
  - `Bidirectional LSTM (32)` + dropout
  - Dense layers -> softmax over `len(WORDS)`
- Optimizer: AdamW
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

## Real-time decoding (`3_traduccion_tiempo_real.py`)
- Maintains a fixed-size sliding window of features
- Predicts once window is filled
- Applies confidence threshold
- Adds predicted tokens into a voting buffer
- Emits only sufficiently stable words (`MIN_VOTES` rule)
- Uses pause logic (`nada_counter`) to prevent repeated same-word spam

---

## 8) Troubleshooting

## Camera does not open
- Verify webcam permissions
- Close other applications using camera

## Empty or poor predictions
- Ensure each word has enough samples
- Improve lighting and framing
- Keep signing style consistent during dataset collection

## Paths not found
- Run scripts from `modeloTutorialFtGemini` directory
- Check `ROOT_PATH` and generated folder structure

## Missing model file in real-time phase
- Train first (`2_entrenamiento_gpu.py`)
- Confirm `MODEL_PATH` points to generated `.keras`

---

## 9) Suggested workflow for experiments

1. Start with a small subset of words (3-6) to validate full loop.
2. Capture balanced samples per class.
3. Train and inspect confusion matrix.
4. Add difficult/confused classes gradually.
5. Re-capture low-quality classes and retrain.

---

## 10) Possible improvements

1. **Refactor to a single CLI**
   - Replace multiple scripts with one CLI entrypoint (e.g., `python main.py capture/train/infer`) and argument flags.

2. **Use project-root-safe path handling**
   - Replace `os.getcwd()` with file-relative path resolution (`Path(__file__).resolve().parent`) to avoid execution-context issues.

3. **Schema and data validation**
   - Add checks for empty samples, invalid sequence lengths, and missing classes before training.

4. **Dedicated test split + reproducible evaluation protocol**
   - Reserve a true held-out test set and report consistent metrics across runs.

5. **Advanced augmentation**
   - Time-warping, frame dropout, mirrored-hand augmentation, and temporal jitter beyond Gaussian noise.

6. **Improve feature engineering**
   - Add velocity/acceleration features from keypoints to improve temporal discrimination.

7. **Model architecture upgrades**
   - Explore Conv1D + BiLSTM hybrid, attention layers, or lightweight Transformers for sequence modeling.

8. **Model/version tracking**
   - Save metadata (vocabulary, hyperparameters, commit hash, timestamp) with each model artifact.

9. **Real-time UX improvements**
   - Add confidence bars with temporal smoothing visualization and optional audio feedback.

10. **Packaging and deployment**
   - Export an inference-only package/app (desktop or API) with clear runtime dependencies.

11. **Performance optimization**
   - Benchmark FPS end-to-end and optimize MediaPipe/model settings for low-latency devices.

12. **Code quality and documentation**
   - Add type hints across all scripts, linting, formatter setup, and unit/integration tests.

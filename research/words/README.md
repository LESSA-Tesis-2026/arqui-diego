# LESSA Word/Phrase Research Pipeline

`research/words` contains the dynamic-sign workflow for collecting word/phrase samples, extracting MediaPipe features, training the temporal model, and validating it with local OpenCV demos. The production API mirrors the feature contract documented here before loading `models/modelo_señas_lstm.keras`.

## File map

```text
research/words/
├── word_config.py                       # Shared labels, paths, MediaPipe extraction, and feature constants
├── collect_h5_sequences.py              # Collect dynamic sequences directly into per-label H5 files
├── collect_raw_videos.py                # Record raw AVI videos when visual review of samples is useful
├── extract_video_keypoints.py           # Convert raw videos into per-label H5 keypoint datasets
├── merge_h5_datasets.py                 # Merge two H5 capture sessions for the same label
├── train_temporal_model.py              # Train the active position+velocity+acceleration model
├── train_position_model_experiment.py   # Train the position-only experiment for comparison/debugging
├── run_temporal_realtime_demo.py        # OpenCV demo for the active temporal-feature model
├── run_position_realtime_demo.py        # OpenCV demo for the position-only experiment
├── requirements.txt                     # Research dependencies for local runs
└── requirements_wsl.txt                 # WSL-specific dependency set
```

Generated folders are local outputs:

```text
keypoint_datasets/  # H5 datasets, one file per word/phrase label
videos/             # Raw captured videos
models/             # Trained .keras/.h5 artifacts
metrics/            # Training charts and reports
data_to_merge/      # Temporary H5 files used by merge_h5_datasets.py
experiments/        # Ad-hoc experiment outputs
.venv/              # Local Python environment
```

## Label contract

Labels are defined in `word_config.py` and must match the trained word model order exactly. Spanish labels are intentional because they represent LESSA output classes and user-facing translation tokens.

`nada` is a rest/no-output state. The live translator uses it to decide when the last emitted word can become eligible for repetition again, so it must stay synchronized with the trained model and runtime state machine.

## Feature contract

Each raw frame produces `306` position features:

```text
pose:          33 landmarks x 4 values = 132
selected face: 16 landmarks x 3 values = 48
left hand:     21 landmarks x 3 values = 63
right hand:    21 landmarks x 3 values = 63
```

The active model uses temporal features. `train_temporal_model.py` concatenates position, first-order velocity, and second-order acceleration, so each frame becomes `918` features. The API must preserve this exact feature order before inference.

## Recommended execution order

### Path A: fastest capture path, direct to H5

Use this path when the goal is to grow the training dataset quickly and raw videos are not needed for manual review.

```bash
cd research/words
python collect_h5_sequences.py
python train_temporal_model.py
python run_temporal_realtime_demo.py
```

### Path B: auditable capture path, videos first

Use this path when you want raw videos available for review before extracting keypoints.

```bash
cd research/words
python collect_raw_videos.py
python extract_video_keypoints.py
python train_temporal_model.py
python run_temporal_realtime_demo.py
```

### Optional: merge H5 datasets

Use `merge_h5_datasets.py` only when two H5 files for the same label need to be combined. Edit the input file names in the script's main block before running it. This manual step is deliberate because merging can duplicate or overwrite sample names if the wrong files are selected.

```bash
cd research/words
python merge_h5_datasets.py
```

## Runtime handoff

Training writes model artifacts to the local `models/` folder inside this workspace. After validating a model, copy the selected artifact to the repository-root runtime folder:

```text
models/modelo_señas_lstm.keras
```

Then run the product path through the FastAPI backend and Next.js frontend. The OpenCV demos are research smoke tests, not the thesis-demo UI.

## Practical capture notes

- Keep the signer centered so the nose-relative normalization has a stable anchor.
- Capture multiple sessions per label when possible; variation in distance, speed, and lighting improves generalization.
- Avoid augmenting validation or test data. Augmentation belongs only in the training split.
- Check generated metrics after training before replacing the runtime artifact.

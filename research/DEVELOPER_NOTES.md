# Research Developer Notes

This document explains the non-obvious decisions in the research code. It is meant for future developers who need to extend the datasets, retrain a model, or compare a new artifact with the runtime application.

## 1. Labels are model contracts

The Spanish word/phrase labels in `research/words/word_config.py` and the alphabet labels in `research/alphabet/alphabet_config.py` are not just display text. They define the class index order used during training and inference.

If a label is renamed, added, removed, or reordered, retrain the corresponding model and update the runtime label order in the API at the same time. A model artifact and a label list with different ordering will produce wrong translations even if the code runs successfully.

## 2. Feature vector contracts

Both research pipelines produce a `306` value position vector per frame:

```text
33 pose landmarks x 4 values      = 132
16 selected face landmarks x 3    = 48
21 left-hand landmarks x 3        = 63
21 right-hand landmarks x 3       = 63
Total                             = 306
```

The active word/phrase model expands this to `918` values per frame by concatenating:

```text
position + velocity + acceleration
306      + 306      + 306          = 918
```

The runtime API must use the same order. Changing the order of pose/face/hand blocks or temporal features requires retraining.

## 3. Nose-relative coordinates

The extraction helpers subtract the nose landmark from pose, selected face, and hand coordinates. This makes samples less sensitive to where the signer stands in the camera frame. If the pose is not detected, the anchor becomes `(0, 0, 0)` and missing landmark groups are filled with zeros.

Zeros are deliberate: training and runtime code use masking/padding to handle missing frames or absent hands. Do not replace missing data with random values.

## 4. Why visibility channels are preserved during augmentation

Pose landmarks include a visibility value every four entries. The training augmentation adds small coordinate noise, then restores every `3::4` visibility channel. Visibility is a MediaPipe confidence signal, not a spatial coordinate, so corrupting it with Gaussian noise would teach the model unrealistic confidence patterns.

## 5. Real-time smoothing and duplicate guards

The OpenCV demo scripts use short voting buffers before accepting a prediction. This reduces flicker from frame-to-frame softmax noise.

Word demos also keep `last_emitted_word` and `rest_counter`:

- `last_emitted_word` prevents repeated output like `hola hola hola` while the same sign remains stable.
- `rest_counter` resets that duplicate guard only after sustained `nada`/no-output frames, allowing the signer to intentionally repeat a word after a pause.

Alphabet demos use a similar duplicate guard for repeated letters. The signer must release or destabilize the current letter before the same letter can be emitted again.

## 6. Generated outputs are not source

The research folders intentionally ignore raw media, H5 datasets, model files, metrics, local environments, and cache files. Source control should contain scripts and documentation; generated outputs should be shared as separate thesis/demo artifacts when needed.

## 7. Recommended extension process

1. Add or adjust labels in the relevant config file.
2. Collect enough samples for the new or changed labels.
3. Extract keypoints using the matching pipeline.
4. Train a new model artifact.
5. Review metrics and smoke-test with the OpenCV demo.
6. Copy the validated artifact to the root `models/` runtime folder.
7. Update API runtime labels/settings and tests if the model contract changed.

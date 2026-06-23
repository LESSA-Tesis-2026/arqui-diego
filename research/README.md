# Research Workspaces

`research/` contains the model-development workflows that produce the runtime artifacts served by the LESSA prototype. The application code in `apps/api` and `apps/web` does not import these scripts directly; it consumes trained model files through the configured runtime paths.

## Workspace map

```text
research/
├── words/       Dynamic word/phrase recognition pipeline
├── alphabet/    Static alphabet recognition pipeline
└── prototypes/  OpenCV-only integration experiments for manual research checks
```

## Naming and language conventions

- Python filenames describe the action they perform: `collect_*`, `extract_*`, `train_*`, `run_*`.
- README workflow sections define the execution order instead of relying on numeric filename prefixes.
- Python identifiers, comments, and developer documentation are in English.
- LESSA labels remain in the language and order used by the trained models. Labels such as `hola`, `buenos_dias`, `mucho_gusto`, and `nada` are data/model contracts, not developer-facing names.

## Generated artifact policy

Research runs create large or machine-specific files. These are local outputs, not source files:

- raw webcam images and videos
- H5 keypoint datasets
- trained `.keras` and `.h5` models
- generated metrics, charts, and reports
- local Python virtual environments
- OS/editor caches

When a trained artifact is ready for the application, copy it into the repository-root `models/` folder for local runtime or Docker mounting. Keep source changes and generated artifacts separate so reviewers can understand the prototype without receiving machine-local outputs.

## Common research flow

1. Collect raw samples or direct H5 sequences.
2. Extract MediaPipe keypoints into H5 datasets when starting from raw media.
3. Train the model for the selected pipeline.
4. Smoke-test the model with the OpenCV demo script.
5. Copy the validated model artifact to the root `models/` folder and run the app through `apps/api` + `apps/web`.

See `research/words/README.md` and `research/alphabet/README.md` for exact commands and artifact locations. See `research/DEVELOPER_NOTES.md` for the model-contract and real-time smoothing decisions that future work must preserve.

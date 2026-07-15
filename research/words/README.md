# Pipeline de investigación de palabras/frases de LESSA

`research/words` contiene el flujo de trabajo de señas dinámicas para recolectar muestras de palabras/frases, extraer características de MediaPipe, entrenar el modelo temporal y validarlo con demos locales de OpenCV. La API de producción replica el contrato de características documentado aquí antes de cargar `models/modelo_señas_lstm.keras`.

## Mapa de archivos

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

Las carpetas generadas son salidas locales:

```text
keypoint_datasets/  # H5 datasets, one file per word/phrase label
videos/             # Raw captured videos
models/             # Trained .keras/.h5 artifacts
metrics/            # Training charts and reports
data_to_merge/      # Temporary H5 files used by merge_h5_datasets.py
experiments/        # Ad-hoc experiment outputs
.venv/              # Local Python environment
```

## Contrato de etiquetas

Las etiquetas se definen en `word_config.py` y deben coincidir exactamente con el orden del modelo de palabras entrenado. Las etiquetas en español son intencionales porque representan las clases de salida de LESSA y los tokens de traducción orientados al usuario.

`nada` es un estado de reposo/sin salida. El traductor en vivo lo usa para decidir cuándo la última palabra emitida puede volver a ser elegible para repetición, así que debe mantenerse sincronizado con el modelo entrenado y la máquina de estados de ejecución.

## Contrato de características

Cada fotograma sin procesar produce `306` características de posición:

```text
pose:          33 landmarks x 4 values = 132
selected face: 16 landmarks x 3 values = 48
left hand:     21 landmarks x 3 values = 63
right hand:    21 landmarks x 3 values = 63
```

El modelo activo usa características temporales. `train_temporal_model.py` concatena la posición, la velocidad de primer orden y la aceleración de segundo orden, de modo que cada fotograma se convierte en `918` características. La API debe preservar este orden exacto de características antes de la inferencia.

## Orden de ejecución recomendado

### Ruta A: la ruta de captura más rápida, directo a H5

Use esta ruta cuando el objetivo sea aumentar rápidamente el dataset de entrenamiento y no se necesiten videos sin procesar para revisión manual.

```bash
cd research/words
python collect_h5_sequences.py
python train_temporal_model.py
python run_temporal_realtime_demo.py
```

### Ruta B: la ruta de captura auditable, videos primero

Use esta ruta cuando quiera tener videos sin procesar disponibles para revisión antes de extraer los keypoints.

```bash
cd research/words
python collect_raw_videos.py
python extract_video_keypoints.py
python train_temporal_model.py
python run_temporal_realtime_demo.py
```

### Opcional: fusionar datasets H5

Use `merge_h5_datasets.py` solo cuando dos archivos H5 de la misma etiqueta necesiten combinarse. Edite los nombres de los archivos de entrada en el bloque principal del script antes de ejecutarlo. Este paso manual es intencional porque la fusión puede duplicar o sobrescribir nombres de muestras si se seleccionan los archivos equivocados.

```bash
cd research/words
python merge_h5_datasets.py
```

## Entrega a ejecución

El entrenamiento escribe los artefactos de modelo en la carpeta `models/` local dentro de este espacio de trabajo. Después de validar un modelo, copie el artefacto seleccionado a la carpeta de ejecución en la raíz del repositorio:

```text
models/modelo_señas_lstm.keras
```

Luego ejecute la ruta de producto a través del backend de FastAPI y el frontend de Next.js. Los demos de OpenCV son pruebas rápidas (smoke tests) de investigación, no la interfaz de la demo de tesis.

## Notas prácticas de captura

- Mantenga al señante centrado para que la normalización relativa a la nariz tenga un ancla estable.
- Capture varias sesiones por etiqueta cuando sea posible; la variación en distancia, velocidad e iluminación mejora la generalización.
- Evite aumentar los datos de validación o de prueba. La aumentación pertenece únicamente al conjunto de entrenamiento.
- Revise las métricas generadas después del entrenamiento antes de reemplazar el artefacto de ejecución.

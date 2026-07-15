# Desarrollo Local

Esta guía cubre el desarrollo diario del prototipo de tesis. Para la entrega y la configuración del revisor, consulte `docs/DELIVERY.md`. Para la arquitectura y los contratos de modelo, consulte `docs/ARCHITECTURE.md`.

## Límites del Repositorio

- `apps/api`: backend de producción de FastAPI.
- `apps/web`: frontend de producción de Next.js.
- `research/words`: espacio de trabajo activo de investigación de palabra/frase.
- `research/alphabet`: espacio de trabajo activo de investigación del alfabeto.
- `research/prototypes`: prototipos de OpenCV para verificaciones de investigación.
- `models`: artefactos de modelo de runtime locales ignorados.
- `docs`: documentación del proyecto.

## Backend

El backend tiene una plantilla de entorno versionada:

```text
apps/api/.env.example
```

Cópiela a un `.env` local al ejecutar fuera de Docker:

```bash
cp apps/api/.env.example apps/api/.env
```

El archivo `.env` local intencionalmente no se versiona. Úselo para valores específicos de la máquina, como los valores absolutos de `LESSA_WORD_MODEL_PATH` y `LESSA_ALPHABET_MODEL_PATH`.

Ejecute el backend de FastAPI desde `apps/api` con `uv`:

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

De forma predeterminada, la API busca los artefactos de modelo en:

```text
models/modelo_señas_lstm.keras
models/modelo_letras.h5
```

El modelo de alfabeto es opcional para el inicio; el modo Alphabet se reporta como no disponible hasta que el archivo exista y se cargue correctamente.

Para el desarrollo local, puede apuntar la API a artefactos explícitos:

```bash
LESSA_WORD_MODEL_PATH="/absolute/path/to/models/modelo_señas_lstm.keras" \
LESSA_ALPHABET_MODEL_PATH="/absolute/path/to/models/modelo_letras.h5" \
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Verificaciones útiles del backend:

```bash
uv run pytest
```

## Frontend

El frontend tiene una plantilla de entorno versionada:

```text
apps/web/.env.example
```

Cópiela a un `.env` local al ejecutar fuera de Docker:

```bash
cp apps/web/.env.example apps/web/.env
```

El archivo `.env` local intencionalmente no se versiona. `NEXT_PUBLIC_API_URL` debe ser alcanzable por el navegador, no solo por el proceso de Next.js.

Ejecute la aplicación web desde `apps/web` con `pnpm`:

```bash
pnpm dev
```

El frontend espera la API en `http://localhost:8000` de forma predeterminada. Anúlelo con:

```bash
NEXT_PUBLIC_API_URL="http://localhost:8000" pnpm dev
```

### Arquitectura del frontend

- Los componentes de presentación de traducción viven bajo `apps/web/src/components/translation`.
- El código de la API del navegador y del ciclo de vida vive bajo `apps/web/src/hooks`.
- Las utilidades de API/WebSocket y el formato de etiquetas viven bajo `apps/web/src/lib`.

### Retroalimentación de voz

La interfaz de traducción usa la API `SpeechSynthesis` del navegador para retroalimentación de voz opcional. Es exclusiva del frontend: el backend emite tokens de texto a través del WebSocket, y el navegador pronuncia solo las emisiones aceptadas (`emitted_token` o `emitted_word`). El modo Words pronuncia palabras o frases completas en español, mientras que el modo Alphabet pronuncia cada letra emitida. Los navegadores no compatibles siguen traduciendo con normalidad y marcan la voz como no disponible en la interfaz de usuario.

Verificaciones útiles del frontend:

```bash
pnpm lint
pnpm build
```

## Cómo agregar una nueva palabra o seña

Este es el flujo típico para quien continúe el modelo de palabras/frases. El detalle de cada script está en `research/words/README.md`; aquí va el mapa general y los puntos donde es fácil equivocarse.

1. **Registrar la etiqueta.** Agregue la nueva palabra al final de la lista `WORDS` en `research/words/word_config.py`. El orden importa: es el contrato de clases del modelo. Agregar al final evita renumerar las etiquetas existentes.
2. **Capturar muestras.** Desde `research/words`, ejecute `python collect_h5_sequences.py` (ruta rápida a H5) o `collect_raw_videos.py` + `extract_video_keypoints.py` (ruta auditable). Capture varias sesiones con variación de distancia, velocidad e iluminación.
3. **Reentrenar.** Ejecute `python train_temporal_model.py`. Revise `metrics/` (matriz de confusión y reporte) antes de dar por bueno el modelo.
4. **Validar rápido.** Pruebe el artefacto con `python run_temporal_realtime_demo.py` (smoke test de OpenCV, no es la demo de tesis).
5. **Sincronizar las etiquetas de la API.** Copie la nueva lista de palabras a `apps/api/app/lessa/labels.py` (`WORD_LABELS`) **en el mismo orden** que en `word_config.py`. Si el orden difiere, la API mostrará etiquetas equivocadas aunque el modelo prediga bien.
6. **Publicar el artefacto.** Copie `research/words/models/modelo_señas_lstm.keras` a la carpeta de ejecución en la raíz: `models/modelo_señas_lstm.keras`.
7. **Verificar de punta a punta.** Levante backend + frontend y confirme que la nueva seña se reconoce y se traduce.

> El proceso para el modelo de alfabeto es análogo, en `research/alphabet` y `ALPHABET_LABELS`.

## Reentrenar y actualizar el modelo

- El entrenamiento escribe el artefacto en `research/words/models/`. Nada se activa en la API hasta que ese archivo se copia a `models/` en la raíz (o se apunta `LESSA_WORD_MODEL_PATH` a otra ruta).
- Si cambia la forma de entrada del modelo (número de landmarks, orden de características, `sequence_length` o el orden de deltas), debe actualizar en conjunto `research/words/word_config.py`, `apps/api/app/lessa/preprocessing.py` y `apps/api/app/core/config.py`, y reentrenar. Ver `docs/ARCHITECTURE.md`, sección "Contrato de características y del modelo".
- La API carga el modelo de forma diferida y reporta el error si el artefacto falta o es incompatible; consulte `GET /api/v1/health` y la metadata del modelo tras un cambio.

## Referencia de parámetros de configuración

Todos se definen en `apps/api/app/core/config.py` y se pueden sobrescribir con el prefijo `LESSA_` (variable de entorno o `.env`). Los marcados con ⚠️ forman parte del contrato del modelo entrenado: cambiarlos sin reentrenar rompe la inferencia.

| Parámetro | Predeterminado | Qué controla |
| --- | --- | --- |
| `sequence_length` ⚠️ | `60` | Fotogramas por muestra que espera la LSTM tras el relleno. |
| `window_size` | `25` | Fotogramas del historial en vivo antes de rellenar a `sequence_length`. |
| `base_feature_length` ⚠️ | `306` | Valores de posición por fotograma (33·4 + 16·3 + 21·3 + 21·3). |
| `use_temporal_features` ⚠️ | `True` | Añade velocidad/aceleración por fotograma. |
| `temporal_delta_order` ⚠️ | `2` | Órdenes de derivada; con `2` la longitud pasa a `918`. |
| `word_confidence_threshold` | `0.65` | Confianza mínima para aceptar una palabra. |
| `alphabet_confidence_threshold` | `0.80` | Confianza mínima para aceptar una letra. |
| `hybrid_motion_threshold` | `0.010` | Umbral de movimiento que separa palabra (dinámica) de alfabeto (estática) en Auto. |
| `settle_seconds` | `3.0` | Tiempo que se debe mantener la seña antes de la primera inferencia. |
| `inference_interval_seconds` | `0.75` | Intervalo mínimo entre inferencias (throttling). |
| `hold_last_reading_seconds` | `2.0` | Cuánto se conserva la última lectura al perder las manos. |
| `voting_buffer_size` / `min_votes` | `10` / `7` | Votación de palabras: se emite una etiqueta al alcanzar los votos. |
| `alphabet_voting_buffer_size` / `alphabet_min_votes` | `5` / `3` | Votación de letras. |
| `max_sentence_words` | `5` | Tokens recientes conservados en la oración devuelta. |
| `pause_reset_frames` | `15` | Fotogramas sin seña válida antes de reiniciar el estado emitido. |

## Docker

Los valores específicos de Docker viven en `docker-compose.yml` porque están acoplados a los puertos de servicio y los montajes de volúmenes.

Anulaciones importantes de Docker:

- `LESSA_WORD_MODEL_PATH=/models/modelo_señas_lstm.keras`, que coincide con el bind mount de models de la raíz del repositorio en el servicio de la API.
- `LESSA_ALPHABET_MODEL_PATH=/models/modelo_letras.h5`, la ruta de contenedor esperada para el artefacto opcional del alfabeto.
- `LESSA_TEMPORAL_DELTA_ORDER=2`, que coincide con la forma de entrada actual del modelo de palabra de `(60, 918)`.
- `LESSA_WINDOW_SIZE=25`, que coincide con la ventana de fotogramas activa antes del relleno (padding) a `60`.
- `LESSA_WORD_CONFIDENCE_THRESHOLD=0.65`, `LESSA_ALPHABET_CONFIDENCE_THRESHOLD=0.80`, `LESSA_HYBRID_MOTION_THRESHOLD=0.010`, `LESSA_SETTLE_SECONDS=3.0`, `LESSA_INFERENCE_INTERVAL_SECONDS=0.75`, `LESSA_VOTING_BUFFER_SIZE=10` y `LESSA_MIN_VOTES=7`.
- `LESSA_CORS_ORIGINS=["http://localhost:3000"]`, que coincide con el origen web publicado.
- `NEXT_PUBLIC_API_URL=http://localhost:8000`, que coincide con el puerto de la API alcanzable desde el navegador.

Ejecute la pila completa desde la raíz del repositorio:

```bash
docker compose up --build
```

La configuración de Compose monta la carpeta de artefactos de modelo de la raíz del repositorio dentro del contenedor de la API:

```text
./models -> /models
```

Mantenga los archivos de runtime ahí cuando estén disponibles:

```text
models/modelo_señas_lstm.keras
models/modelo_letras.h5
```

Luego abra la aplicación web en `http://localhost:3000`. La API está disponible en `http://localhost:8000/api/v1/health`.

En Apple Silicon, si los wheels de Linux de TensorFlow o MediaPipe fallan para las builds nativas de ARM, ejecute:

```bash
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose up --build
```

## Archivos Generados y Ruido Local

Los archivos locales ignorados incluyen `.env`, `.venv`, `.next`, `node_modules`, `models/`, cachés, datasets/videos/métricas generados y `apps/web/next-env.d.ts`. Estos archivos son salidas de runtime o específicas de la máquina, en lugar de archivos de entrega de código fuente.

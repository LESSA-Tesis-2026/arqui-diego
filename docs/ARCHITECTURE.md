# Arquitectura

Este repositorio contiene el prototipo de tesis para la traducción en vivo de LESSA a español. El producto ejecutable es una experiencia de navegador respaldada por un servicio de inferencia con FastAPI. Los espacios de trabajo de investigación explican cómo se producen los artefactos de modelo, pero no son importados por las aplicaciones de producción en runtime.

## Límites del Repositorio

```text
apps/api/                 FastAPI model-serving backend
apps/web/                 Next.js browser translation experience
research/words/           Active word/phrase research workspace
research/alphabet/        Active static alphabet research workspace
research/prototypes/      Earlier OpenCV hybrid-translation prototypes
models/                   Local runtime model artifacts, ignored by Git
docs/                     Architecture, delivery, and development docs
docker-compose.yml        Local full-stack runtime
```

Los espacios de trabajo de investigación usan nombres de carpetas y archivos en inglés y son activos de desarrollo de modelos, no paquetes de aplicación de producción. Las aplicaciones de producción consumen los artefactos entrenados a través de rutas de modelo configuradas en lugar de importar scripts de investigación.

## Convención de Idioma

El prototipo es para hablantes de LESSA y español, por lo que la salida de dominio orientada al español es intencional.

| Área | Convención |
| --- | --- |
| Identificadores de código, módulos, carpetas, campos de API | Inglés |
| Comentarios y documentación de desarrollador | Español |
| Texto de interfaz orientado al usuario | Español |
| Etiquetas de modelo y clases de dataset | Preservar el contrato de etiquetas entrenadas |

Ejemplos como `hola`, `buenos_dias`, `mi_nombre_es`, `nada` y las clases del alfabeto son etiquetas de modelo. No las renombre ni las reordene a menos que un artefacto de modelo futuro se entrene con un nuevo contrato de etiquetas.

## Flujo de Runtime

```text
Browser camera
  -> apps/web frame capture
  -> WS /api/v1/translate/stream
  -> apps/api frame decoding
  -> MediaPipe Holistic landmarks
  -> LESSA feature vectors
  -> word/alphabet model inference
  -> voting and stabilization
  -> Spanish text and UI feedback
```

1. El frontend solicita permiso de cámara y previsualiza el stream localmente.
2. Mientras la traducción está activa, los fotogramas se limitan (throttle) y se codifican como data URLs JPEG.
3. El backend recibe los mensajes de fotogramas a través de WebSocket.
4. MediaPipe extrae los puntos de referencia (landmarks) de pose, rostro seleccionado y manos.
5. El backend construye la forma exacta de características esperada por los artefactos de modelo entrenados.
6. El modo Auto enruta las señas en movimiento al reconocimiento de palabra y las señas estáticas al reconocimiento de alfabeto.
7. Los búferes de votación y las ventanas de asentamiento reducen el parpadeo y las emisiones inestables repetidas.
8. El frontend renderiza el texto en español, el estado actual, la confianza de la predicción, el historial y la retroalimentación de voz opcional del navegador.

## Estructura del Backend

```text
apps/api/app/
├── api/routes/            FastAPI HTTP/WebSocket route handlers
├── core/config.py         Environment-driven settings
├── lessa/labels.py        Stable model label ordering
├── lessa/preprocessing.py Frame decoding, MediaPipe, feature vectors
├── lessa/runtimes.py      Lazy Keras model loading and metadata
├── lessa/session.py       Per-WebSocket translation state
├── lessa/inference.py     Mode routing, prediction, voting, responses
└── lessa/schemas.py       Pydantic API/WebSocket schemas
```

Las rutas deben permanecer ligeras. Coloque la validación de solicitudes y el ciclo de vida del socket en los módulos de rutas; coloque el comportamiento del modelo en los módulos `app.lessa`.

## Estructura del Frontend

```text
apps/web/src/
├── components/translation/  Translation UI pieces
├── hooks/                   Browser API and socket lifecycle hooks
├── lib/api.ts               REST API utilities
├── lib/websocket.ts         WebSocket message utilities/types
└── lib/labels.ts            Model-label to Spanish-display formatting
```

La lógica de la API del navegador vive en los hooks:

- `useCamera`: permiso de cámara, propiedad del stream, limpieza.
- `useFrameStreaming`: limitación (throttling) de fotogramas, captura de canvas, backpressure del socket.
- `useTranslationSocket`: disponibilidad del modelo, ciclo de vida del socket, estado de traducción, manejo de reinicio, voz de emisión estable.
- `useSpeechSynthesis`: integración de la Web Speech API segura para la hidratación.

Los componentes de presentación reciben props normalizadas y no deben ser propietarios de recursos del navegador de bajo nivel.

## Contrato de Artefactos de Modelo

Los artefactos de runtime se cargan desde `models/` de forma predeterminada:

```text
models/modelo_señas_lstm.keras
models/modelo_letras.h5
```

Estos archivos son intencionalmente ignorados por Git. Pueden copiarse localmente o montarse en Docker.

Contratos importantes:

- Las etiquetas del modelo de palabra provienen de `apps/api/app/lessa/labels.py` en orden exacto.
- Las etiquetas del modelo de alfabeto son `ABCDEFGHIKLMNOPQRSTUVWXY`; `J` y `Z` están ausentes porque el clasificador de alfabeto actual se basa en fotogramas estáticos.
- El modelo de palabra actual espera una longitud de secuencia de `60` y una longitud de características de `918`.
- La longitud de características de posición base es `306`.
- Las características temporales añaden deltas de primer y segundo orden después de las características de posición.
- El orden del índice de rostro seleccionado difiere entre los modelos de palabra y alfabeto y debe permanecer estable.

## Contrato de características y del modelo

Esta es la parte más frágil del sistema: los vectores que produce la API en tiempo de ejecución deben coincidir **exactamente** con los que se usaron para entrenar el artefacto. Si no coinciden, el modelo carga sin error pero predice mal. Un desarrollador que continúe este trabajo debe entender estas cifras antes de tocar el preprocesamiento o la configuración.

**Vector de posición por fotograma (`base_feature_length = 306`).** Se construye en `apps/api/app/lessa/preprocessing.py` concatenando, en este orden:

| Bloque | Puntos × valores | Subtotal | Rango de índices |
| --- | --- | --- | --- |
| Pose | 33 × 4 (x, y, z, visibility) | 132 | 0–131 |
| Rostro seleccionado | 16 × 3 (x, y, z) | 48 | 132–179 |
| Mano izquierda | 21 × 3 | 63 | 180–242 |
| Mano derecha | 21 × 3 | 63 | 243–305 |
| **Total** | | **306** | |

- Todas las coordenadas se anclan restando la posición de la nariz (pose landmark 0), de modo que el modelo aprende movimiento relativo a la persona y no la posición absoluta en la cámara.
- Las manos empiezan en el índice `180`; por eso `motion_score` mide el movimiento solo a partir de ese offset para decidir el modo en Auto.
- Los índices de rostro (`WORD_FACE_INDICES` vs `ALPHABET_FACE_INDICES`) difieren entre modelos y deben permanecer estables.

**Características temporales (`feature_length = 918`).** Cuando `use_temporal_features = True` y `temporal_delta_order = 2`, cada fotograma se enriquece concatenando posición + velocidad (1ª derivada) + aceleración (2ª derivada): `306 × (1 + 2) = 918`. Esta transformación se hace en `build_sequence_features` (API) y debe ser idéntica a `compute_deltas` del script de entrenamiento `research/words/train_temporal_model.py`.

**Longitud de secuencia (`sequence_length = 60`) frente a ventana (`window_size = 25`).** El modelo espera secuencias de `60` fotogramas. En vivo, la sesión mantiene una ventana deslizante de solo `25` fotogramas (movimiento reciente) y `pad_sequence` la rellena con ceros hasta `60`; el modelo ignora el relleno gracias a su capa `Masking`.

En resumen, deben mantenerse alineados: los índices de landmarks y su orden, `base_feature_length`, `temporal_delta_order`, `sequence_length`, el orden de etiquetas en `labels.py` y la misma lógica de deltas en API y entrenamiento. Cambiar cualquiera de ellos obliga a reentrenar el modelo.

## Archivos Generados y Locales

Los artefactos locales/generados son salidas de runtime o específicas de la máquina:

- archivos `.env`
- `.venv`, `node_modules`, `.next`, cachés
- `apps/web/next-env.d.ts`
- archivos de runtime bajo `models/`
- datasets, videos, archivos H5 y métricas generados, a menos que se acepten explícitamente como evidencia de tesis

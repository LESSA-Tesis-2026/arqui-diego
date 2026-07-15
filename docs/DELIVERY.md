# Guía de Entrega de Tesis

Esta guía es la lista de verificación de entrega para ejecutar y revisar el prototipo de tesis de traducción de LESSA a español.

## Alcance de la Entrega

Incluido en la entrega:

- `apps/api`: backend de FastAPI para el servido de modelos y el streaming de traducción en vivo.
- `apps/web`: frontend de Next.js para la experiencia de traducción en navegador.
- `research/words`: espacio de trabajo activo de investigación de palabra/frase.
- `research/alphabet`: espacio de trabajo activo de investigación del alfabeto.
- `research/prototypes`: prototipos de OpenCV para verificaciones de investigación.
- `docs`: notas de configuración, arquitectura, entrega y trabajo futuro.
- `docker-compose.yml`: runtime local de pila completa.

No forma parte de la entrega de código fuente:

- Archivos generados como `apps/web/next-env.d.ts`.
- Dependencias locales, cachés, `.env` y entornos virtuales.
- Artefactos de modelo de runtime, a menos que se distribuyan explícitamente a través de un artefacto de lanzamiento separado.

## Artefactos de Modelo

Coloque los artefactos de modelo de runtime en la raíz del repositorio:

```text
models/modelo_señas_lstm.keras
models/modelo_letras.h5
```

El modelo de palabra es requerido para el reconocimiento de palabras. El modelo de alfabeto es opcional al iniciar; si falta, la API reporta el modo Alphabet como no disponible.

## Backend Local

```bash
cd apps/api
cp .env.example .env
uv sync
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Verificación de salud:

```text
http://localhost:8000/api/v1/health
```

Metadatos del modelo:

```text
http://localhost:8000/api/v1/model/info
```

## Frontend Local

```bash
cd apps/web
cp .env.example .env
pnpm install
pnpm dev
```

Abra:

```text
http://localhost:3000
```

El navegador debe poder alcanzar `NEXT_PUBLIC_API_URL`, cuyo valor predeterminado es `http://localhost:8000`.

## Runtime de Docker

Desde la raíz del repositorio:

```bash
docker compose up --build
```

Luego abra:

```text
http://localhost:3000
```

Docker monta `./models` dentro del contenedor de la API en `/models`.

En Apple Silicon, si los wheels de TensorFlow o MediaPipe fallan para las builds nativas de ARM, use:

```bash
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose up --build
```

## Lista de Verificación

Ejecute antes de la entrega de tesis:

```bash
cd apps/api
uv run pytest
```

```bash
cd apps/web
pnpm lint
pnpm build
```

Después de las verificaciones, inspeccione el estado de Git. Los archivos ignorados que se esperan solo localmente pueden incluir `.env`, `.venv`, `.next`, `node_modules`, `models/` y `apps/web/next-env.d.ts`. No deben aparecer como modificaciones versionadas.

## Trabajo Futuro

### Agregar o cambiar señas

Las adiciones requieren primero trabajo de investigación/modelo. Actualice los datasets, entrene un nuevo modelo, y solo entonces actualice las etiquetas de runtime. El orden de las etiquetas de runtime debe coincidir exactamente con el artefacto entrenado.

### Reemplazar artefactos de modelo

Al reemplazar modelos, verifique:

- el orden de las etiquetas
- la longitud de secuencia de entrada esperada
- la longitud de características
- la selección de puntos de referencia (landmarks) del rostro
- el orden de los deltas temporales
- los umbrales de confianza y votación

Actualice la documentación y las pruebas con el nuevo contrato.

### Ajuste de umbrales

Las configuraciones relevantes del backend incluyen:

- `LESSA_WORD_CONFIDENCE_THRESHOLD`
- `LESSA_ALPHABET_CONFIDENCE_THRESHOLD`
- `LESSA_HYBRID_MOTION_THRESHOLD`
- `LESSA_SETTLE_SECONDS`
- `LESSA_INFERENCE_INTERVAL_SECONDS`
- `LESSA_VOTING_BUFFER_SIZE`
- `LESSA_MIN_VOTES`
- `LESSA_ALPHABET_VOTING_BUFFER_SIZE`
- `LESSA_ALPHABET_MIN_VOTES`

Umbrales más bajos pueden hacer que la demo se sienta más rápida, pero pueden emitir señas incorrectas. Umbrales más altos pueden reducir los errores, pero pueden sentirse menos responsivos.

### Despliegue

Este prototipo está listo para local/Docker. Un despliegue de producción aún necesitaría decisiones explícitas sobre TLS, distribución de artefactos de modelo, dimensionamiento de cómputo, privacidad, autenticación si se expone más allá de una red de demostración, observabilidad y pruebas de compatibilidad de cámara/navegador.

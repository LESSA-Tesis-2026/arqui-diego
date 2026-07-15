# LESSA Translation

Prototipo de tesis para la traducción en vivo de LESSA a español. El proyecto combina un backend de servido de modelos con FastAPI, una experiencia de traducción en navegador con Next.js, espacios de trabajo activos de desarrollo de modelos, documentación compartida y orquestación con Docker para el runtime local. El prototipo admite inferencia híbrida de palabra/frase y alfabeto.

## Mapa del Repositorio

- `apps/api`: backend de FastAPI. Es responsable del preprocesamiento de fotogramas, la carga de modelos, la inferencia híbrida, la estabilización y el contrato público REST/WebSocket.
- `apps/web`: frontend de Next.js. Es responsable del acceso a la cámara, los controles Auto/Words/Alphabet, el estado en vivo, la salida de traducción al español y la retroalimentación de voz opcional del navegador.
- `research/words`: espacio de trabajo activo de investigación/desarrollo de modelos para el LSTM de palabra/frase. Las aplicaciones de producción consumen el artefacto `.keras` entrenado de este flujo de trabajo en lugar de importar estos scripts directamente.
- `research/alphabet`: espacio de trabajo activo de investigación/desarrollo de modelos para el clasificador estático del alfabeto. La aplicación puede cargar su artefacto `.h5` entrenado cuando esté disponible.
- `research/prototypes`: prototipos de traducción híbrida con OpenCV para verificaciones de investigación.
- `models`: carpeta local de artefactos de runtime. Es ignorada por Git y debe contener los archivos de modelo para la ejecución local/Docker.
- `docs`: documentación de arquitectura, entrega y desarrollo a nivel de proyecto.
- `docker-compose.yml`: runtime local de pila completa para la aplicación web y la API.


## Idioma y Etiquetas de Modelo

El prototipo es para hablantes de LESSA y español:

- El texto orientado al usuario permanece en español.
- Las clases de LESSA y las etiquetas de modelo entrenadas permanecen estables, por ejemplo `hola`, `buenos_dias`, `mi_nombre_es` y `nada`.
- La documentación y los comentarios de código están en español.
- Los identificadores de código, nombres de módulos y campos de API se mantienen en inglés, a menos que representen texto en español orientado al usuario o etiquetas de modelo.

## Cómo se Conecta

El navegador captura fotogramas de la cámara en `apps/web` y los transmite (stream) a la API a través de un WebSocket:

```text
apps/web -> WS /api/v1/translate/stream -> apps/api
```

La API decodifica cada fotograma, extrae los puntos de referencia (landmarks) de mano/cuerpo una sola vez, construye los vectores de características de palabra y alfabeto, y elige el reconocedor activo. El modo Auto enruta las señas en movimiento al LSTM de palabra y las señas estáticas al clasificador del alfabeto; la interfaz de usuario también puede forzar el modo Words o Alphabet. La API espera una breve ventana de asentamiento antes de la inferencia y limita las predicciones para evitar parpadeo. El frontend renderiza el estado de reconocimiento, la confianza, las predicciones recientes y el texto progresivo en español. Cuando la síntesis de voz del navegador está disponible, el frontend puede pronunciar las emisiones aceptadas: palabras/frases completas en modo Words y letras individuales en modo Alphabet.

Los modelos entrenados son artefactos de runtime. No son construidos (build) por las aplicaciones web o API. El modelo de palabra sigue siendo requerido para el reconocimiento de palabras; el modelo de alfabeto es opcional y su modo se marca como no disponible hasta que exista el artefacto `.h5`.

## Flujo de Runtime

1. El usuario abre la aplicación web en `http://localhost:3000`.
2. El navegador solicita permiso de cámara.
3. El frontend se conecta al WebSocket de la API en `http://localhost:8000/api/v1/translate/stream`.
4. El frontend envía fotogramas codificados mientras la traducción está activa.
5. La API preprocesa los fotogramas y ejecuta la inferencia de palabra o alfabeto según el modo seleccionado y la puntuación de movimiento.
6. La API envía eventos de traducción híbrida de vuelta al navegador.
7. El frontend muestra la traducción actual y el historial de la sesión.
8. Si la retroalimentación de voz está habilitada y es compatible con el navegador, los tokens emitidos aceptados se pronuncian con la API SpeechSynthesis del navegador.

## Documentación

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): límites del repositorio, flujo de runtime, estructura de backend/frontend, contratos de modelo.
- [`docs/DELIVERY.md`](docs/DELIVERY.md): configuración de la demo de tesis, lista de verificación, política de archivos generados, notas de trabajo futuro.
- [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md): comandos de desarrollo local y Docker.
- [`research/README.md`](research/README.md): límites del espacio de trabajo de investigación y política de artefactos generados.

## Desarrollo Local

Verificaciones del backend:

```bash
cd apps/api
uv run pytest
```

Verificaciones del frontend:

```bash
cd apps/web
pnpm lint
pnpm build
```

Consulte [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) para la configuración completa.

## Docker

Ejecute la pila completa desde la raíz del repositorio:

```bash
docker compose up --build
```

Luego abra:

```text
http://localhost:3000
```

El endpoint de salud de la API está disponible en:

```text
http://localhost:8000/api/v1/health
```

## Entorno

Cada aplicación tiene una plantilla de entorno versionada:

- `apps/api/.env.example`
- `apps/web/.env.example`

Los archivos `.env` locales intencionalmente no se versionan. La API usa `LESSA_WORD_MODEL_PATH` y `LESSA_ALPHABET_MODEL_PATH` para los dos artefactos de runtime. Los valores específicos de Docker viven en `docker-compose.yml` porque están vinculados a los puertos publicados y los montajes de volúmenes.

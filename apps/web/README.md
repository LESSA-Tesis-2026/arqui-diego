# LESSA Translation Web

Experiencia inmersiva en el navegador para la traducción en tiempo real de LESSA a español. El frontend gestiona el acceso a la cámara, los controles de traducción en vivo, la salida progresiva en español y la retroalimentación de voz opcional del navegador mientras transmite fotogramas al backend de FastAPI.

## Estructura

```text
src/
├── app/                         Next.js app route and layout
├── components/translation/      Translation UI components
├── components/ui/               Shared UI primitives
├── hooks/                       Browser API and socket lifecycle hooks
└── lib/                         API, WebSocket, label-formatting utilities
```

Módulos clave:

- `components/translation/translation-experience.tsx`: orquesta la experiencia.
- `hooks/use-camera.ts`: permiso de cámara, control del stream, limpieza.
- `hooks/use-frame-streaming.ts`: captura de canvas, limitación (throttling), backpressure de WebSocket.
- `hooks/use-translation-socket.ts`: disponibilidad de la API, ciclo de vida del WebSocket, estado de traducción, manejo del reinicio.
- `hooks/use-speech-synthesis.ts`: integración de la Web Speech API segura frente a la hydration.
- `lib/labels.ts`: mapea las etiquetas estables del modelo a texto de visualización en español.

## Idioma y Etiquetas

El texto de cara al usuario está en español. Las etiquetas del modelo son contratos estables del backend/modelo y no deben renombrarse en el código del frontend. Use `lib/labels.ts` cuando una etiqueta del modelo necesite formato de visualización.

## Tech

- Next.js
- React
- TypeScript
- Tailwind CSS
- shadcn/ui-style primitives
- `pnpm` para la gestión de packages

## Entorno

`.env.example` es la fuente de verdad versionada para la configuración del frontend. Cree un `.env` local a partir de él cuando ejecute la aplicación web fuera de Docker:

```bash
cp .env.example .env
```

`.env` es únicamente local. Los valores específicos de Docker se declaran en el `docker-compose.yml` de la raíz.

Variables:

- `NEXT_PUBLIC_API_URL`: URL del backend de FastAPI. Su valor por defecto es `http://localhost:8000` para el desarrollo local y Docker, porque el navegador accede a la API a través del puerto mapeado en el host.

## Instalación

```bash
pnpm install
```

## Ejecución

Inicie primero el backend y luego ejecute:

```bash
pnpm dev
```

Abra:

```text
http://localhost:3000
```

## Flujo Principal

- Active el permiso de la cámara.
- Inicie la traducción en vivo.
- Transmita los fotogramas de la cámara a `WS /api/v1/translate/stream`.
- Muestre el estado de reconocimiento y el texto progresivo en español.
- Reproduzca por voz las emisiones estables aceptadas cuando la retroalimentación de voz esté habilitada y sea compatible con el navegador.
- Pause o borre la sesión de traducción actual.

## Notas sobre las APIs del Navegador

- El navegador requiere permiso de cámara para la traducción en vivo.
- El streaming de fotogramas está limitado (throttling) y descarta fotogramas si hay bytes de WebSocket en cola, evitando predicciones obsoletas y retrasadas.
- La síntesis de voz se detecta después de la hydration para evitar discrepancias de render entre servidor y cliente.
- El audio se activa únicamente a partir de emisiones estables del backend (`emitted_token` o `emitted_word`), nunca a partir de predicciones inestables a nivel de fotograma.

## Archivos Generados

`next-env.d.ts` es generado por Next.js y se ignora/no se versiona de forma intencional. Puede cambiar entre los builds de desarrollo y producción, por lo que los archivos fuente no deben depender de editarlo a mano.

## Verificaciones

```bash
pnpm lint
pnpm build
```

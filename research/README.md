# Espacios de trabajo de investigación

`research/` contiene los flujos de desarrollo de modelos que producen los artefactos de ejecución que sirve el prototipo de LESSA. El código de la aplicación en `apps/api` y `apps/web` no importa estos scripts directamente; consume los archivos de modelos entrenados a través de las rutas de ejecución configuradas.

## Mapa del espacio de trabajo

```text
research/
├── words/       Dynamic word/phrase recognition pipeline
├── alphabet/    Static alphabet recognition pipeline
└── prototypes/  OpenCV-only integration experiments for manual research checks
```

## Convenciones de nomenclatura e idioma

- Los nombres de archivo de Python describen la acción que realizan: `collect_*`, `extract_*`, `train_*`, `run_*`.
- Las secciones de flujo de trabajo del README definen el orden de ejecución en lugar de depender de prefijos numéricos en los nombres de archivo.
- Los identificadores de Python se mantienen en inglés; los comentarios y la documentación para desarrolladores están en español.
- Las etiquetas de LESSA se conservan en el idioma y el orden que usan los modelos entrenados. Etiquetas como `hola`, `buenos_dias`, `mucho_gusto` y `nada` son contratos de datos/modelo, no nombres orientados al desarrollador.

## Política de artefactos generados

Las ejecuciones de investigación crean archivos grandes o específicos de la máquina. Estos son salidas locales, no archivos fuente:

- imágenes y videos de webcam sin procesar
- datasets de keypoints en H5
- modelos entrenados `.keras` y `.h5`
- métricas, gráficas y reportes generados
- entornos virtuales locales de Python
- cachés del sistema operativo o del editor

Cuando un artefacto entrenado esté listo para la aplicación, cópielo en la carpeta `models/` en la raíz del repositorio para la ejecución local o para el montaje con Docker. Mantenga separados los cambios de código fuente y los artefactos generados para que quienes revisen puedan entender el prototipo sin recibir salidas locales de la máquina.

## Flujo de investigación común

1. Recolecte muestras sin procesar o secuencias H5 directas.
2. Extraiga los keypoints de MediaPipe a datasets H5 cuando parta de medios sin procesar.
3. Entrene el modelo para el pipeline seleccionado.
4. Realice una prueba rápida (smoke-test) del modelo con el script de demo de OpenCV.
5. Copie el artefacto de modelo validado a la carpeta `models/` en la raíz y ejecute la aplicación a través de `apps/api` + `apps/web`.

Consulte `research/words/README.md` y `research/alphabet/README.md` para los comandos exactos y las ubicaciones de los artefactos. Consulte `research/DEVELOPER_NOTES.md` para las decisiones sobre el contrato del modelo y el suavizado en tiempo real que el trabajo futuro debe preservar.

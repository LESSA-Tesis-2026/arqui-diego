# Notas de desarrollo de investigación

Este documento explica las decisiones no evidentes del código de investigación. Está pensado para futuros desarrolladores que necesiten ampliar los datasets, reentrenar un modelo o comparar un nuevo artefacto con la aplicación de ejecución.

## 1. Las etiquetas son contratos del modelo

Las etiquetas de palabra/frase en español en `research/words/word_config.py` y las etiquetas del alfabeto en `research/alphabet/alphabet_config.py` no son solo texto para mostrar. Definen el orden de los índices de clase que se usa durante el entrenamiento y la inferencia.

Si una etiqueta se renombra, se agrega, se elimina o se reordena, reentrene el modelo correspondiente y actualice el orden de etiquetas de ejecución en la API al mismo tiempo. Un artefacto de modelo y una lista de etiquetas con un orden distinto producirán traducciones incorrectas incluso si el código se ejecuta correctamente.

## 2. Contratos del vector de características

Ambos pipelines de investigación producen un vector de posición de `306` valores por fotograma:

```text
33 pose landmarks x 4 values      = 132
16 selected face landmarks x 3    = 48
21 left-hand landmarks x 3        = 63
21 right-hand landmarks x 3       = 63
Total                             = 306
```

El modelo de palabra/frase activo expande esto a `918` valores por fotograma al concatenar:

```text
position + velocity + acceleration
306      + 306      + 306          = 918
```

La API de ejecución debe usar el mismo orden. Cambiar el orden de los bloques de pose/rostro/mano o de las características temporales requiere reentrenar.

## 3. Coordenadas relativas a la nariz

Los ayudantes de extracción restan el punto de referencia (landmark) de la nariz de las coordenadas de pose, del rostro seleccionado y de las manos. Esto hace que las muestras sean menos sensibles a la posición donde se ubica el señante dentro del fotograma de la cámara. Si no se detecta la pose, el ancla pasa a ser `(0, 0, 0)` y los grupos de landmarks faltantes se rellenan con ceros.

Los ceros son intencionales: el código de entrenamiento y de ejecución usa enmascaramiento/relleno (masking/padding) para manejar fotogramas faltantes o manos ausentes. No reemplace los datos faltantes con valores aleatorios.

## 4. Por qué los canales de visibilidad se preservan durante la aumentación

Los landmarks de pose incluyen un valor de visibilidad cada cuatro entradas. La aumentación de entrenamiento agrega un pequeño ruido de coordenadas y luego restaura cada canal de visibilidad `3::4`. La visibilidad es una señal de confianza de MediaPipe, no una coordenada espacial, así que corromperla con ruido gaussiano le enseñaría al modelo patrones de confianza poco realistas.

## 5. Suavizado en tiempo real y protecciones contra duplicados

Los scripts de demo de OpenCV usan búferes de votación cortos antes de aceptar una predicción. Esto reduce el parpadeo causado por el ruido de softmax entre fotogramas.

Los demos de palabras también mantienen `last_emitted_word` y `rest_counter`:

- `last_emitted_word` evita salidas repetidas como `hola hola hola` mientras la misma seña permanece estable.
- `rest_counter` reinicia esa protección contra duplicados solo después de fotogramas sostenidos de `nada`/sin salida, lo que permite que el señante repita intencionalmente una palabra tras una pausa.

Los demos del alfabeto usan una protección contra duplicados similar para letras repetidas. El señante debe soltar o desestabilizar la letra actual antes de que la misma letra pueda emitirse de nuevo.

## 6. Las salidas generadas no son fuente

Las carpetas de investigación ignoran intencionalmente los medios sin procesar, los datasets H5, los archivos de modelo, las métricas, los entornos locales y los archivos de caché. El control de versiones debe contener scripts y documentación; las salidas generadas deben compartirse como artefactos de tesis/demo separados cuando sea necesario.

## 7. Proceso de extensión recomendado

1. Agregue o ajuste etiquetas en el archivo de configuración correspondiente.
2. Recolecte suficientes muestras para las etiquetas nuevas o modificadas.
3. Extraiga los keypoints usando el pipeline correspondiente.
4. Entrene un nuevo artefacto de modelo.
5. Revise las métricas y haga una prueba rápida (smoke-test) con el demo de OpenCV.
6. Copie el artefacto validado a la carpeta de ejecución `models/` en la raíz.
7. Actualice las etiquetas/ajustes de ejecución de la API y las pruebas si cambió el contrato del modelo.

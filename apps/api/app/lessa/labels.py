"""Contratos estables de etiquetas del modelo LESSA.

Estas cadenas no son identificadores de código generales. Son el orden exacto de clases que
usan los artefactos del modelo entrenado, por lo que cambiar la ortografía o el orden requiere
reentrenar o una migración explícita de los metadatos del modelo.
"""

WORD_LABELS: list[str] = [
    "hola",
    "buenos_dias",
    "gracias",
    "mucho gusto",
    "mi_nombre_es",
    "cuidate",
    "nada",
    "buenas tardes",
    "buenas noches",
    "como estas",
    "cual es tu nombre",
    "permiso",
    "adios",
    "perdon",
    "otra vez",
    "por favor",
    "duda",
    "nos vemos luego",
    "por que",
    "si",
    "no",
    "talvez",
    "no se",
]

# La J y la Z están ausentes intencionalmente porque este clasificador de alfabeto se basa
# en fotogramas estáticos; esas letras de LESSA son típicamente señas de movimiento.
ALPHABET_LABELS: list[str] = list("ABCDEFGHIKLMNOPQRSTUVWXY")

# Alias retrocompatible para llamadores internos antiguos durante las refactorizaciones.
LABELS = WORD_LABELS

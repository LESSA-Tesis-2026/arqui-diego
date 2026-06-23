"""Stable LESSA model label contracts.

These strings are not general code identifiers. They are the exact class order used
by the trained model artifacts, so changing spelling or ordering requires retraining
or an explicit model-metadata migration.
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

# J and Z are intentionally absent because this alphabet classifier is static-frame
# based; those LESSA letters are typically motion signs.
ALPHABET_LABELS: list[str] = list("ABCDEFGHIKLMNOPQRSTUVWXY")

# Backwards-compatible alias for older internal callers during refactors.
LABELS = WORD_LABELS

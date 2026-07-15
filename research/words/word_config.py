"""Constantes compartidas y ayudantes de MediaPipe para el pipeline de investigación de palabras/frases.

Mantenga el orden de las etiquetas y la extracción de características alineados con el modelo de palabras entrenado.
Las cadenas de etiquetas en español son salidas del modelo y no deben renombrarse a la ligera.
"""

import os

import cv2
import mediapipe as mp
import numpy as np

# Contrato de secuencia y de características usado por la LSTM de palabras/frases entrenada.
MAX_FRAMES = 60

# Los modelos actuales usan un vector compacto de 306 valores:
# 33 pose landmarks x 4 + 16 selected face landmarks x 3 + 21 left-hand x 3 + 21 right-hand x 3.
SELECTED_FACE_INDICES = [
    61, 291, 0, 17, 13, 14,      # Labios y comisuras de la boca
    33, 133, 362, 263,           # Comisuras de los ojos
    70, 63, 105, 336, 296, 334   # Cejas
]
LENGTH_KEYPOINTS = 306

# Rutas del espacio de trabajo. Son relativas a la carpeta para que los scripts puedan ejecutarse desde este directorio de investigación.
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_PATH, "keypoint_datasets")
MODEL_FOLDER_PATH = os.path.join(ROOT_PATH, "models")
METRICS_FOLDER = os.path.join(ROOT_PATH, "metrics")
MODEL_PATH = os.path.join(MODEL_FOLDER_PATH, "modelo_señas_lstm.keras")
VIDEOS_FOLDER = os.path.join(ROOT_PATH, "videos")

# Etiquetas estables del modelo. Estas etiquetas en español/LESSA son parte del contrato del artefacto entrenado.
WORDS = [
    "hola", "buenos_dias", "gracias", "mucho gusto", "mi_nombre_es", "cuidate",
    "nada", "buenas tardes", "buenas noches", "como estas", "cual es tu nombre",
    "permiso", "adios", "perdon", "otra vez", "por favor", "duda", "nos vemos luego",
    "por que", "si", "no", "talvez", "no se",
]


def mediapipe_detection(image, model):
    """Procesa un fotograma de OpenCV a través de un modelo de MediaPipe y devuelve la salida BGR más los landmarks.

    OpenCV lee fotogramas en BGR, mientras que MediaPipe espera entrada en RGB. La bandera writeable se
    desactiva durante el procesamiento porque MediaPipe puede omitir una copia interna cuando el
    fotograma es de solo lectura."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def extract_keypoints(results):
    """Extrae el vector de características de 306 valores anclado en la nariz que usan los modelos de palabras."""
    if results.pose_landmarks:
        anchor_x = results.pose_landmarks.landmark[0].x
        anchor_y = results.pose_landmarks.landmark[0].y
        anchor_z = results.pose_landmarks.landmark[0].z
    else:
        anchor_x, anchor_y, anchor_z = 0.0, 0.0, 0.0

    if results.pose_landmarks:
        pose = np.array([[res.x - anchor_x, res.y - anchor_y, res.z - anchor_z, res.visibility]
                         for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33 * 4)

    if results.left_hand_landmarks:
        left_hand = np.array([[res.x - anchor_x, res.y - anchor_y, res.z - anchor_z]
                              for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        left_hand = np.zeros(21 * 3)

    if results.right_hand_landmarks:
        right_hand = np.array([[res.x - anchor_x, res.y - anchor_y, res.z - anchor_z]
                               for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        right_hand = np.zeros(21 * 3)

    if results.face_landmarks:
        face = np.array([[results.face_landmarks.landmark[i].x - anchor_x,
                          results.face_landmarks.landmark[i].y - anchor_y,
                          results.face_landmarks.landmark[i].z - anchor_z]
                         for i in SELECTED_FACE_INDICES]).flatten()
    else:
        face = np.zeros(len(SELECTED_FACE_INDICES) * 3)

    return np.concatenate([pose, face, left_hand, right_hand])


def create_folder_if_not_exists(path):
    """Crea una carpeta del espacio de trabajo cuando un script está por escribir salidas de investigación generadas."""
    if not os.path.exists(path):
        os.makedirs(path)

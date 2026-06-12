import os
import cv2
import numpy as np
import mediapipe as mp

# SETTINGS
MAX_FRAMES = 60 # Longitud máxima para el padding (puede ajustare si se hacen señas muy largas)
#LENGTH_KEYPOINTS = 1662 # 33*4 (pose) + 468*3 (face) + 21*3 (lh) + 21*3 (rh)
# 16 puntos faciales clave: Boca, Ojos y Cejas
SELECTED_FACE_INDICES = [
    61, 291, 0, 17, 13, 14,      # Labios y comisuras
    33, 133, 362, 263,           # Esquinas de los ojos
    70, 63, 105, 336, 296, 334   # Cejas
]

# 33*4 (pose) + 16*3 (cara seleccionada) + 21*3 (lh) + 21*3 (rh)
LENGTH_KEYPOINTS = 306

# PATHS
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_PATH, "data_h5")
MODEL_FOLDER_PATH = os.path.join(ROOT_PATH, "models")
METRICS_FOLDER = os.path.join(ROOT_PATH, "metricas_modelo")
MODEL_PATH = os.path.join(MODEL_FOLDER_PATH, "modelo_señas_lstm.keras")
VIDEOS_FOLDER = os.path.join(ROOT_PATH, "videos")

# DICCIONARIO DE PALABRAS
WORDS = ["hola", "buenos_dias", "gracias", "mucho gusto", "mi_nombre_es", "cuidate", "nada", "buenas tardes", "buenas noches", "como estas", "cual es tu nombre", "permiso", "adios", "perdon", "otra vez", "por favor", "duda", "nos vemos luego", "por que", "si", "no", "talvez", "no se"] 

# ["hola", "buenos_dias", "gracias", "mucho gusto", "mi_nombre_es", "cuidate", "nada", "buenas tardes", "buenas noches", "como estas", "cual es tu nombre", "permiso", "adios", "perdon", "otra vez", "por favor", "duda", "nos vemos luego", "por que", "si", "no", "talvez", "no se"]

# FUNCIONES MEDIAPIPE
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# funcion modificada para obtener coordenadas relativas
def extract_keypoints(results):
    # 1. Definir la Nariz (Pose Landmark 0) como nuestro "Ancla" central.
    # Si MediaPipe no detecta el cuerpo, el ancla es 0,0,0
    if results.pose_landmarks:
        ancla_x = results.pose_landmarks.landmark[0].x
        ancla_y = results.pose_landmarks.landmark[0].y
        ancla_z = results.pose_landmarks.landmark[0].z
    else:
        ancla_x, ancla_y, ancla_z = 0.0, 0.0, 0.0

    # 2. Extraer POSE restando el ancla a cada punto (Movimiento puro)
    if results.pose_landmarks:
        pose = np.array([[res.x - ancla_x, res.y - ancla_y, res.z - ancla_z, res.visibility] 
                         for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33 * 4)

    # 3. Extraer MANOS y ROSTRO restando el ancla
    if results.left_hand_landmarks:
        lh = np.array([[res.x - ancla_x, res.y - ancla_y, res.z - ancla_z] 
                       for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21 * 3)

    if results.right_hand_landmarks:
        rh = np.array([[res.x - ancla_x, res.y - ancla_y, res.z - ancla_z] 
                       for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3)
        
    if results.face_landmarks:
        face = np.array([[results.face_landmarks.landmark[i].x - ancla_x, 
                          results.face_landmarks.landmark[i].y - ancla_y, 
                          results.face_landmarks.landmark[i].z - ancla_z] 
                         for i in SELECTED_FACE_INDICES]).flatten()
    else:
        face = np.zeros(len(SELECTED_FACE_INDICES) * 3)
        
    return np.concatenate([pose, face, lh, rh])

def create_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
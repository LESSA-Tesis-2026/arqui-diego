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
ROOT_PATH = os.getcwd()
DATA_PATH = os.path.join(ROOT_PATH, "data_h5")
MODEL_FOLDER_PATH = os.path.join(ROOT_PATH, "models")
METRICS_FOLDER = os.path.join(ROOT_PATH, "metricas_modelo")
MODEL_PATH = os.path.join(MODEL_FOLDER_PATH, "modelo_señas_lstm.keras")

# DICCIONARIO DE PALABRAS
WORDS = ["hola", "buenos_dias", "gracias", "mucho gusto", "mi_nombre_es", "cuidate", "nada"] # Agrega aquí todas tus palabras

# FUNCIONES MEDIAPIPE
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # Extracción selectiva del rostro
    if results.face_landmarks:
        face = np.array([[results.face_landmarks.landmark[i].x, 
                          results.face_landmarks.landmark[i].y, 
                          results.face_landmarks.landmark[i].z] 
                         for i in SELECTED_FACE_INDICES]).flatten()
    else:
        face = np.zeros(len(SELECTED_FACE_INDICES) * 3)
        
    return np.concatenate([pose, face, lh, rh])

def create_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
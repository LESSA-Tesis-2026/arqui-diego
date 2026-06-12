import os
import cv2
import numpy as np
import mediapipe as mp

# 1. CLASES DEL ALFABETO
ALPHABET = list("ABCDEFGHIKLMNOPQRSTUVWXY") 

# 2. RUTAS DE DIRECTORIOS
DATASET_FOLDER = "dataset_alfabeto"
DATA_H5_FOLDER = "data_h5_alfabeto"
MODEL_FOLDER = "modelo_alfabeto"
METRICS_FOLDER = os.path.join(MODEL_FOLDER, "metricas")
MODEL_PATH = os.path.join(MODEL_FOLDER, "modelo_letras.h5")

# 3. PUNTOS FACIALES SELECCIONADOS (Igual que en palabras)
SELECTED_FACE_INDICES = [33, 133, 362, 263, 1, 61, 291, 199, 94, 0, 11, 13, 14, 15, 16, 17]
LENGTH_KEYPOINTS = 306 # 132(Pose) + 48(Cara) + 63(Mano Izq) + 63(Mano Der)

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def mediapipe_detection(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), results

def extract_keypoints(results):
    """Extrae las 306 coordenadas relativas a la nariz (Ancla 0,0,0)"""
    if results.pose_landmarks:
        ancla_x = results.pose_landmarks.landmark[0].x
        ancla_y = results.pose_landmarks.landmark[0].y
        ancla_z = results.pose_landmarks.landmark[0].z
    else:
        ancla_x, ancla_y, ancla_z = 0.0, 0.0, 0.0

    if results.pose_landmarks:
        pose = np.array([[res.x - ancla_x, res.y - ancla_y, res.z - ancla_z, res.visibility] 
                         for res in results.pose_landmarks.landmark]).flatten()
    else: pose = np.zeros(33 * 4)

    if results.left_hand_landmarks:
        lh = np.array([[res.x - ancla_x, res.y - ancla_y, res.z - ancla_z] 
                       for res in results.left_hand_landmarks.landmark]).flatten()
    else: lh = np.zeros(21 * 3)

    if results.right_hand_landmarks:
        rh = np.array([[res.x - ancla_x, res.y - ancla_y, res.z - ancla_z] 
                       for res in results.right_hand_landmarks.landmark]).flatten()
    else: rh = np.zeros(21 * 3)
        
    if results.face_landmarks:
        face = np.array([[results.face_landmarks.landmark[i].x - ancla_x, 
                          results.face_landmarks.landmark[i].y - ancla_y, 
                          results.face_landmarks.landmark[i].z - ancla_z] 
                         for i in SELECTED_FACE_INDICES]).flatten()
    else: face = np.zeros(len(SELECTED_FACE_INDICES) * 3)
        
    return np.concatenate([pose, face, lh, rh])
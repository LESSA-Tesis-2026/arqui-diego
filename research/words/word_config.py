"""Shared constants and MediaPipe helpers for the word/phrase research pipeline.

Keep label order and feature extraction aligned with the trained word model.
Spanish label strings are model outputs and should not be renamed casually.
"""

import os

import cv2
import mediapipe as mp
import numpy as np

# Sequence and feature contract used by the trained word/phrase LSTM.
MAX_FRAMES = 60

# Current models use a compact 306-value vector:
# 33 pose landmarks x 4 + 16 selected face landmarks x 3 + 21 left-hand x 3 + 21 right-hand x 3.
SELECTED_FACE_INDICES = [
    61, 291, 0, 17, 13, 14,      # Lips and mouth corners
    33, 133, 362, 263,           # Eye corners
    70, 63, 105, 336, 296, 334   # Eyebrows
]
LENGTH_KEYPOINTS = 306

# Workspace paths. These are folder-relative so scripts can be run from this research directory.
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_PATH, "keypoint_datasets")
MODEL_FOLDER_PATH = os.path.join(ROOT_PATH, "models")
METRICS_FOLDER = os.path.join(ROOT_PATH, "metrics")
MODEL_PATH = os.path.join(MODEL_FOLDER_PATH, "modelo_señas_lstm.keras")
VIDEOS_FOLDER = os.path.join(ROOT_PATH, "videos")

# Stable model labels. These Spanish/LESSA labels are part of the trained artifact contract.
WORDS = [
    "hola", "buenos_dias", "gracias", "mucho gusto", "mi_nombre_es", "cuidate",
    "nada", "buenas tardes", "buenas noches", "como estas", "cual es tu nombre",
    "permiso", "adios", "perdon", "otra vez", "por favor", "duda", "nos vemos luego",
    "por que", "si", "no", "talvez", "no se",
]


def mediapipe_detection(image, model):
    """Run one OpenCV frame through a MediaPipe model and return BGR output plus landmarks.

    OpenCV reads BGR frames, while MediaPipe expects RGB input. The writeable flag is
    turned off during processing because MediaPipe can skip an internal copy when the
    frame is read-only."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def extract_keypoints(results):
    """Extract the 306-value nose-anchored feature vector used by word models."""
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
    """Create a workspace folder when a script is about to write generated research output."""
    if not os.path.exists(path):
        os.makedirs(path)

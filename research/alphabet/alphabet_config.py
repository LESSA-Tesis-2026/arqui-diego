"""Shared constants and MediaPipe helpers for the alphabet research pipeline.

The alphabet label order must stay aligned with `modelo_letras.h5`.
"""

import os

import cv2
import mediapipe as mp
import numpy as np

# Static alphabet labels. J and Z are intentionally absent because those LESSA signs usually require motion.
ALPHABET = list("ABCDEFGHIKLMNOPQRSTUVWXY")

# Workspace paths. Generated data/model folders are ignored by Git.
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = os.path.join(ROOT_PATH, "raw_images")
DATA_H5_FOLDER = os.path.join(ROOT_PATH, "keypoint_datasets")
MODEL_FOLDER = os.path.join(ROOT_PATH, "trained_model")
METRICS_FOLDER = os.path.join(MODEL_FOLDER, "metrics")
MODEL_PATH = os.path.join(MODEL_FOLDER, "modelo_letras.h5")

# Alphabet model face landmarks differ from the word model; keep this order stable for artifact compatibility.
SELECTED_FACE_INDICES = [33, 133, 362, 263, 1, 61, 291, 199, 94, 0, 11, 13, 14, 15, 16, 17]
LENGTH_KEYPOINTS = 306  # 132 pose + 48 selected face + 63 left hand + 63 right hand.


def create_folder_if_not_exists(folder_path):
    """Create a workspace folder when a script is about to write generated research output."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def mediapipe_detection(image, model):
    """Run one OpenCV frame through a MediaPipe model and return BGR output plus landmarks.

    OpenCV uses BGR and MediaPipe uses RGB, so this helper centralizes the color-space
    conversion for all alphabet research scripts."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), results


def extract_keypoints(results):
    """Extract the 306-value nose-anchored feature vector used by the static alphabet model."""
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

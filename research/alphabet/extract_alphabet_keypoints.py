"""Convert raw alphabet images into 306-value MediaPipe keypoint H5 datasets."""

import cv2
import numpy as np
import h5py
import mediapipe as mp
import os
from alphabet_config import *

def extract_alphabet_keypoints():
    """Convert alphabet images into one 306-value H5 dataset per letter.

    Frames without detected hands are skipped because they would teach the classifier
    that missing hands are valid examples of a letter."""
    create_folder_if_not_exists(DATA_H5_FOLDER)
    mp_holistic = mp.solutions.holistic

    with mp_holistic.Holistic(min_detection_confidence=0.5, static_image_mode=True) as holistic:
        for letter in ALPHABET:
            letter_folder = os.path.join(DATASET_FOLDER, letter)
            if not os.path.exists(letter_folder): continue

            h5_file_path = os.path.join(DATA_H5_FOLDER, f"{letter}.h5")
            image_files = [f for f in os.listdir(letter_folder) if f.endswith('.jpg')]

            if not image_files: continue
            print(f"\n--- EXTRACTING LANDMARKS: Letter {letter} ---")

            with h5py.File(h5_file_path, 'a') as hf:
                existing_datasets = list(hf.keys())

                for img_file in image_files:
                    dataset_name = os.path.splitext(img_file)[0]
                    if dataset_name in existing_datasets: continue

                    img_path = os.path.join(letter_folder, img_file)
                    image = cv2.imread(img_path)
                    if image is None: continue

                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image_rgb)

                    # Discard images where no hand landmarks were detected.
                    if not results.left_hand_landmarks and not results.right_hand_landmarks:
                        continue

                    keypoints = extract_keypoints(results)
                    # Store one static 306-value landmark vector per image.
                    hf.create_dataset(dataset_name, data=keypoints)

            print(f"[OK] Letter {letter} saved to .h5")

if __name__ == "__main__":
    extract_alphabet_keypoints()

"""Collect dynamic LESSA word/phrase samples directly into H5 sequence files.

Use this script when raw video review is not needed. It appends new samples to the
existing per-label H5 file, which lets a capture session resume without deleting
previously recorded sequences.
"""

import cv2
import numpy as np
import h5py
import mediapipe as mp
import os
from word_config import *

def draw_custom_keypoints(image, results):
    """Draw the same compact landmark overlay used by the word/phrase capture tools.

    Only selected face points are drawn because the full face mesh hides the hands and
    makes it harder to review whether a capture is usable."""
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    # Draw only the selected face landmarks to keep the capture overlay readable.
    if results.face_landmarks:
        h, w, _ = image.shape
        for idx in SELECTED_FACE_INDICES:
            landmark = results.face_landmarks.landmark[idx]
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (cx, cy), 3, (0, 255, 255), -1)

    # Draw pose and hand landmarks with MediaPipe's standard connections.
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

def collect_dynamic_h5_sequences(word, target_samples=60):
    """Append new dynamic-sign sequences for one label into its H5 dataset.

    Each sample stores a variable-length list of 306-value frame vectors. Training
    pads/truncates later, so the capture step preserves the natural sign duration."""

    # The rest class needs extra examples because it acts as the transition/no-output state.
    if word.lower() == "nada":
        target_samples = int(target_samples * 1.75)
        print(f"\n[INFO] Class 'nada' detected. Target adjusted automatically a {target_samples} samples.")

    create_folder_if_not_exists(DATA_PATH)
    file_path = os.path.join(DATA_PATH, f"{word}.h5")

    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Append mode lets interrupted sessions resume without losing previously captured samples.
        with h5py.File(file_path, 'a') as hf:

            existing_samples = len(hf.keys())

            # Skip labels that already reached the requested target sample count.
            if existing_samples >= target_samples:
                print(f"[*] Word '{word.upper()}' already has {existing_samples} samples. Skipping...")
                cap.release()
                cv2.destroyAllWindows()
                return

            # Compute only the missing samples so the H5 file can be grown incrementally.
            samples_to_record = target_samples - existing_samples
            print(f"\n--- COLLECTING SAMPLES FOR: {word.upper()} ---")
            print(f"Existing samples: {existing_samples} | Remaining: {samples_to_record} to reach target of {target_samples}.")

            for i in range(samples_to_record):
                # Use the absolute sample index to avoid overwriting existing datasets.
                current_sample_idx = existing_samples + i
                sequence_data = []
                recording = False

                while True:
                    ret, frame = cap.read()
                    if not ret: break

                    image, results = mediapipe_detection(frame, holistic)
                    draw_custom_keypoints(image, results)

                    # Show progress against the total requested sample count.
                    cv2.putText(image, f"Label: {word} | Sample: {current_sample_idx + 1}/{target_samples}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    if recording:
                        cv2.putText(image, "RECORDING... (press 's' to stop)", (10, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        keypoints = extract_keypoints(results)
                        sequence_data.append(keypoints)
                    else:
                        cv2.putText(image, "Press 'r' to start capture", (10, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    cv2.imshow('Data Capture', image)
                    key = cv2.waitKey(10) & 0xFF

                    if key == ord('r') and not recording:
                        recording = True
                    elif key == ord('s') and recording:
                        recording = False

                        # Store the sequence with the stable absolute sample index.
                        dataset_name = f"sample_{current_sample_idx}"
                        hf.create_dataset(dataset_name, data=np.array(sequence_data))
                        print(f"Sample {current_sample_idx + 1} saved with {len(sequence_data)} frames.")
                        break
                    elif key == ord('q'):
                        print("\nRecording interrupted by the user.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Global target: existing samples are kept and only missing samples are appended.
    # ACTUAL: 80
    TARGET_SAMPLES = 80

    for word in WORDS:
        collect_dynamic_h5_sequences(word, target_samples=TARGET_SAMPLES)

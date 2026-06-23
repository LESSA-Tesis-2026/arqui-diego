"""Collect raw webcam images for each static LESSA alphabet class.

Run this before keypoint extraction. Existing images are counted first so capture
sessions can be resumed without replacing previous samples.
"""

import cv2
import os
import time
from alphabet_config import *

def draw_landmarks(image, results):
    """Draw MediaPipe landmarks on the preview frame so the operator can check hand visibility."""
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )

    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )

    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
        )

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
        )

def collect_alphabet_images(letter, target_images=200, delay_seconds=0.1):
    """Collect raw static-letter images for one alphabet class.

    The script saves clean frames while showing an annotated preview. Captures are
    throttled by `delay_seconds` so samples are not near-identical duplicates."""
    letter_folder = os.path.join(DATASET_FOLDER, letter)
    create_folder_if_not_exists(letter_folder)

    existing_images = len([f for f in os.listdir(letter_folder) if f.endswith('.jpg')])
    if existing_images >= target_images:
        print(f"[*] Letter '{letter}' already has {existing_images} images. Skipping...")
        return

    images_to_capture = target_images - existing_images
    print(f"\n--- COLLECTING IMAGES FOR: {letter} ---")

    cap = cv2.VideoCapture(0)
    recording = False
    paused = False
    last_capture_time = time.time()
    current_idx = existing_images

    with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while current_idx < target_images:
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)

            display_frame = frame.copy()
            draw_landmarks(display_frame, results)

            cv2.putText(
                display_frame,
                f"Letter: {letter} | Captures: {current_idx}/{target_images}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

            if recording and paused:
                cv2.putText(
                    display_frame,
                    "PAUSED - reposition and press SPACE to continue",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                )
            elif recording:
                cv2.putText(
                    display_frame,
                    "CAPTURING... (SPACE pauses)",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

                remaining = max(0.0, delay_seconds - (time.time() - last_capture_time))
                cv2.putText(
                    display_frame,
                    f"Next capture in: {remaining:.1f}s",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

                # Timer gate controls the capture cadence.
                if time.time() - last_capture_time > delay_seconds:
                    img_path = os.path.join(letter_folder, f"img_{current_idx}.jpg")
                    cv2.imwrite(img_path, frame)  # Guardamos el frame LIMPIO
                    current_idx += 1
                    last_capture_time = time.time()

                    # Visual flash confirms that a frame was captured.
                    cv2.rectangle(
                        display_frame,
                        (0, 0),
                        (frame.shape[1], frame.shape[0]),
                        (255, 255, 255),
                        -1,
                    )
            else:
                cv2.putText(
                    display_frame,
                    "Press 'r' to start auto-capture",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow('LESSA Alphabet Capture', display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):
                recording = True
                paused = False
                last_capture_time = time.time()
            elif key == ord(' '):
                if recording:
                    paused = not paused
                    if not paused:
                        last_capture_time = time.time()
            elif key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    TARGET_IMAGES = 100 # Recommended for static signs
    DELAY_SECONDS = 1.0 # Delay between captured frames.

    create_folder_if_not_exists(DATASET_FOLDER)
    for letter in ALPHABET:
        collect_alphabet_images(letter, target_images=TARGET_IMAGES, delay_seconds=DELAY_SECONDS)

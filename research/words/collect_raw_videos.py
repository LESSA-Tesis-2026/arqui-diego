"""Record raw webcam videos for each LESSA word/phrase label.

Use this script when samples should be visually reviewed before keypoint
extraction. Run `extract_video_keypoints.py` afterward to build H5 datasets.
"""

import cv2
import os
import mediapipe as mp
from word_config import *


def draw_custom_keypoints(image, results):
    """Draw a lightweight overlay for operator feedback while saving clean raw frames."""
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    if results.face_landmarks:
        h, w, _ = image.shape
        for idx in SELECTED_FACE_INDICES:
            landmark = results.face_landmarks.landmark[idx]
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (cx, cy), 3, (0, 255, 255), -1)

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

def create_video_folders():
    """Ensure every configured word/phrase label has a destination folder for AVI files."""
    if not os.path.exists(VIDEOS_FOLDER):
        os.makedirs(VIDEOS_FOLDER)
    for word in WORDS:
        word_path = os.path.join(VIDEOS_FOLDER, word)
        if not os.path.exists(word_path):
            os.makedirs(word_path)

def collect_raw_videos(word, target_samples=80):
    """Record raw AVI samples for one label without embedding the landmark overlay.

    The preview window is annotated for the operator, but the saved file receives the
    original camera frame so later keypoint extraction sees realistic input."""
    if word.lower() == "nada":
        target_samples = int(target_samples * 1.75)
        print(f"\n[INFO] Class 'nada' detected. Target adjusted a {target_samples} videos.")

    word_folder = os.path.join(VIDEOS_FOLDER, word)

    # Count existing AVI videos so interrupted capture sessions can resume safely.
    existing_videos = len([f for f in os.listdir(word_folder) if f.endswith('.avi')])

    if existing_videos >= target_samples:
        print(f"[*] The label '{word.upper()}' already has {existing_videos} videos. Skipping...")
        return

    samples_to_record = target_samples - existing_videos
    print(f"\n--- COLLECTING VIDEOS FOR: {word.upper()} ---")
    print(f"Existing videos: {existing_videos} | Remaining: {samples_to_record}")

    cap = cv2.VideoCapture(0)
    # Read the camera resolution so saved videos match the raw frame size.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30.0 # Standard frames per second.
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # Compatible and efficient codec.

    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for i in range(samples_to_record):
            current_sample_idx = existing_videos + i
            recording = False
            out = None

            while True:
                ret, frame = cap.read()
                if not ret: break

                # Process the frame only for live visual feedback.
                image, results = mediapipe_detection(frame, holistic)
                draw_custom_keypoints(image, results)

                cv2.putText(image, f"Label: {word} | Video: {current_sample_idx + 1}/{target_samples}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                if recording:
                    cv2.putText(image, "RECORDING... (press 's' to stop)", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # Important: save the raw frame, not the annotated preview image.
                    out.write(frame)
                else:
                    cv2.putText(image, "Press 'r' to start capture", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('LESSA Video Capture', image)
                key = cv2.waitKey(10) & 0xFF

                if key == ord('r') and not recording:
                    recording = True
                    # Initialize the video writer when recording starts.
                    video_path = os.path.join(word_folder, f"sample_{current_sample_idx}.avi")
                    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

                elif key == ord('s') and recording:
                    recording = False
                    out.release() # Close and persist the video file.
                    print(f"Video {current_sample_idx + 1} saved successfully.")
                    break

                elif key == ord('q'):
                    print("\nRecording interrupted by the user.")
                    if out is not None: out.release()
                    cap.release()
                    cv2.destroyAllWindows()
                    return

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    TARGET_SAMPLES = 80
    create_video_folders()
    for word in WORDS:
        collect_raw_videos(word, target_samples=TARGET_SAMPLES)

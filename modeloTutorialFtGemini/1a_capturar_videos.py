import os
import time

import cv2
import mediapipe as mp

from config import *


def draw_custom_keypoints(image, results):
    """Draw custom points/landmarks for on-screen visualization."""
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    if results.face_landmarks:
        h, w, _ = image.shape
        for idx in SELECTED_FACE_INDICES:
            landmark = results.face_landmarks.landmark[idx]
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (cx, cy), 3, (0, 255, 255), -1)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )


def create_video_folders():
    """Create output folders per word if they do not exist."""
    if not os.path.exists(VIDEOS_FOLDER):
        os.makedirs(VIDEOS_FOLDER)
    for word in WORDS:
        word_path = os.path.join(VIDEOS_FOLDER, word)
        if not os.path.exists(word_path):
            os.makedirs(word_path)


def _format_seconds(value: float) -> str:
    """Format seconds based on configured UI precision."""
    return f"{value:.{DISPLAY_TIMER_DECIMALS}f}"


def _render_status_overlay(image, word: str, sample_idx: int, target_samples: int, phase: str, elapsed: float = 0.0):
    """Render status text for each capture phase.

    Args:
        image: Frame shown on screen.
        word: Current word label.
        sample_idx: Sample index (0-based).
        target_samples: Total video target for the word.
        phase: Current state-machine phase.
        elapsed: Elapsed time in the current phase.
    """
    cv2.putText(
        image,
        f"Palabra: {word} | Video: {sample_idx + 1}/{target_samples}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
    )

    if phase == "IDLE":
        message = "Press 'r' to start capture"
        color = (0, 255, 0)
    elif phase == "COUNTDOWN":
        remaining = max(0.0, PRE_RECORD_COUNTDOWN_SECONDS - elapsed)
        message = f"Starts in {_format_seconds(remaining)}s"
        color = (0, 255, 255)
    elif phase == "RECORDING":
        message = f"RECORDING {_format_seconds(elapsed)}/{_format_seconds(RECORD_DURATION_SECONDS)}s"
        color = (0, 0, 255)
    else:
        message = "Unknown state"
        color = (255, 255, 255)

    cv2.putText(image, message, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def capture_videos(word, target_samples=100):
    """Capture videos per word with pre-start countdown and fixed auto-stop.

    Per-sample flow:
    1. `IDLE`: waits for `r` key.
    2. `COUNTDOWN`: shows configurable countdown.
    3. `RECORDING`: records video and stops automatically at target duration.

    Args:
        word: Word label to capture.
        target_samples: Total target number of videos for the word.

    Raises:
        ValueError: If configured recording duration is invalid (<= 0).
    """
    if RECORD_DURATION_SECONDS <= 0:
        raise ValueError("RECORD_DURATION_SECONDS must be greater than 0.")

    if word.lower() == "nada":
        target_samples = int(target_samples * 1.75)
        print(f"\n[INFO] 'nada' class detected. Target adjusted to {target_samples} videos.")

    word_folder = os.path.join(VIDEOS_FOLDER, word)
    existing_videos = len([f for f in os.listdir(word_folder) if f.endswith(".avi")])

    if existing_videos >= target_samples:
        print(f"[*] Word '{word.upper()}' already has {existing_videos} videos. Skipping...")
        return

    samples_to_record = target_samples - existing_videos
    print(f"\n--- COLLECTING VIDEOS FOR: {word.upper()} ---")
    print(f"Existing videos: {existing_videos} | Remaining: {samples_to_record}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for i in range(samples_to_record):
            current_sample_idx = existing_videos + i
            phase = "IDLE"
            phase_start_ts = 0.0
            recording_start_ts = 0.0
            out = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Camera read failed. Ending capture.")
                    if out is not None:
                        out.release()
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                image, results = mediapipe_detection(frame, holistic)
                draw_custom_keypoints(image, results)
                now = time.perf_counter()

                # Use a monotonic clock to avoid drift from system clock adjustments.
                if phase == "COUNTDOWN":
                    countdown_elapsed = now - phase_start_ts
                    if countdown_elapsed >= PRE_RECORD_COUNTDOWN_SECONDS:
                        video_path = os.path.join(word_folder, f"sample_{current_sample_idx}.avi")
                        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                        phase = "RECORDING"
                        recording_start_ts = now
                        _render_status_overlay(image, word, current_sample_idx, target_samples, phase, 0.0)
                    else:
                        _render_status_overlay(
                            image, word, current_sample_idx, target_samples, phase, countdown_elapsed
                        )

                elif phase == "RECORDING":
                    recording_elapsed = now - recording_start_ts

                    # Save raw frames to avoid contaminating the output video with overlays/landmarks.
                    out.write(frame)
                    _render_status_overlay(
                        image, word, current_sample_idx, target_samples, phase, recording_elapsed
                    )

                    if recording_elapsed >= RECORD_DURATION_SECONDS:
                        out.release()
                        out = None
                        print(
                            f"Video {current_sample_idx + 1} saved successfully "
                            f"({_format_seconds(RECORD_DURATION_SECONDS)}s)."
                        )
                        break

                else:  # IDLE
                    _render_status_overlay(image, word, current_sample_idx, target_samples, phase, 0.0)

                cv2.imshow("Captura de Videos de LESSA", image)
                key = cv2.waitKey(10) & 0xFF

                if key == ord("q"):
                    print("\nRecording interrupted by user.")
                    if out is not None:
                        out.release()
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                if key == ord("r") and phase == "IDLE":
                    phase = "COUNTDOWN"
                    phase_start_ts = time.perf_counter()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    META_MUESTRAS = 100
    create_video_folders()
    for word in WORDS:
        capture_videos(word, target_samples=META_MUESTRAS)

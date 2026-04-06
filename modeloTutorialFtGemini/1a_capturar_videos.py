import os
import time

import cv2
import mediapipe as mp

from config import *
from capture_utils import (
    build_capture_plan,
    create_video_folders,
    draw_state_border,
    format_seconds,
    render_status_overlay,
)


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
    planned_indices, extra_indices = build_capture_plan(word_folder, target_samples)
    completed_videos = target_samples - len(planned_indices)

    if not planned_indices:
        print(f"[*] Word '{word.upper()}' is complete ({completed_videos}/{target_samples}). Skipping...")
        if extra_indices:
            print(f"[*] Found extra indices outside target range: {extra_indices}")
        return

    print(f"\n--- COLLECTING VIDEOS FOR: {word.upper()} ---")
    print(f"Valid in-range videos: {completed_videos}/{target_samples}")
    print(f"Missing indices to capture: {planned_indices}")
    if extra_indices:
        print(f"Extra indices (not counted toward target): {extra_indices}")

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
        for capture_order, current_sample_idx in enumerate(planned_indices, start=1):
            phase = "IDLE"
            phase_start_ts = 0.0
            recording_start_ts = 0.0
            out = None
            progress_idx = capture_order - 1

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
                draw_state_border(image, phase)

                # Use a monotonic clock to avoid drift from system clock adjustments.
                if phase == "COUNTDOWN":
                    countdown_elapsed = now - phase_start_ts
                    if countdown_elapsed >= PRE_RECORD_COUNTDOWN_SECONDS:
                        video_path = os.path.join(word_folder, f"sample_{current_sample_idx}.avi")
                        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                        phase = "RECORDING"
                        recording_start_ts = now
                        render_status_overlay(
                            image,
                            word,
                            progress_idx,
                            len(planned_indices),
                            phase,
                            PRE_RECORD_COUNTDOWN_SECONDS,
                            RECORD_DURATION_SECONDS,
                            0.0,
                        )
                    else:
                        render_status_overlay(
                            image,
                            word,
                            progress_idx,
                            len(planned_indices),
                            phase,
                            PRE_RECORD_COUNTDOWN_SECONDS,
                            RECORD_DURATION_SECONDS,
                            countdown_elapsed,
                        )

                elif phase == "RECORDING":
                    recording_elapsed = now - recording_start_ts

                    # Save raw frames to avoid contaminating the output video with overlays/landmarks.
                    out.write(frame)
                    render_status_overlay(
                        image,
                        word,
                        progress_idx,
                        len(planned_indices),
                        phase,
                        PRE_RECORD_COUNTDOWN_SECONDS,
                        RECORD_DURATION_SECONDS,
                        recording_elapsed,
                    )

                    if recording_elapsed >= RECORD_DURATION_SECONDS:
                        out.release()
                        out = None
                        print(
                            f"Video {current_sample_idx + 1} saved successfully "
                            f"({format_seconds(RECORD_DURATION_SECONDS)}s)."
                        )
                        break

                else:  # IDLE
                    render_status_overlay(
                        image,
                        word,
                        capture_order - 1,
                        len(planned_indices),
                        phase,
                        PRE_RECORD_COUNTDOWN_SECONDS,
                        RECORD_DURATION_SECONDS,
                        0.0,
                    )

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
    create_video_folders(VIDEOS_FOLDER, WORDS)
    for word in WORDS:
        capture_videos(word, target_samples=META_MUESTRAS)

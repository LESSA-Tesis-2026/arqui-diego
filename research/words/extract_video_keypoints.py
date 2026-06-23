"""Convert raw word/phrase videos into MediaPipe keypoint H5 datasets.

The extractor reuses `word_config.extract_keypoints` so the research dataset
matches the same nose-relative feature contract expected by training and runtime.
"""

import cv2
import numpy as np
import h5py
import mediapipe as mp
import os
from word_config import *


def extract_video_keypoints():
    """Batch-convert raw word/phrase videos into per-label H5 keypoint sequences.

    Existing dataset keys are skipped, which allows the extractor to be run multiple
    times after interrupted capture sessions without duplicating samples."""
    # Ensure the output folder exists before writing H5 datasets.
    create_folder_if_not_exists(DATA_PATH)
    mp_holistic = mp.solutions.holistic

    # Start MediaPipe once and reuse it across all videos for faster batch processing.
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        for word in WORDS:
            word_folder = os.path.join(VIDEOS_FOLDER, word)
            if not os.path.exists(word_folder):
                print(f"[*] The video folder for '{word}' does not exist. Skipping...")
                continue

            h5_file_path = os.path.join(DATA_PATH, f"{word}.h5")
            video_files = [f for f in os.listdir(word_folder) if f.endswith('.avi')]

            if not video_files:
                continue

            print(f"\n--- EXTRACTING VIDEO LANDMARKS: {word.upper()} ---")

            # Open the H5 file in append mode so processing can resume without overwriting existing samples.
            with h5py.File(h5_file_path, 'a') as hf:
                existing_datasets = list(hf.keys())

                for video_file in video_files:
                    # Convert 'sample_0.avi' into the dataset key 'sample_0'.
                    dataset_name = os.path.splitext(video_file)[0]

                    # Skip videos that have already been converted into H5 datasets.
                    if dataset_name in existing_datasets:
                        continue

                    video_path = os.path.join(word_folder, video_file)
                    cap = cv2.VideoCapture(video_path)
                    sequence_data = []

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break # End of video.

                        # Run the raw frame through MediaPipe.
                        image, results = mediapipe_detection(frame, holistic)

                        # Use the shared config helper, which normalizes coordinates relative to the nose.
                        keypoints = extract_keypoints(results)
                        sequence_data.append(keypoints)

                    cap.release()

                    # Persist the extracted keypoint sequence in the H5 file.
                    if sequence_data:
                        hf.create_dataset(dataset_name, data=np.array(sequence_data))
                        print(f"Processed: {dataset_name} | Length: {len(sequence_data)} frames")
                    else:
                        print(f"WARNING: No keypoint information was detected in {video_file}")

if __name__ == "__main__":
    print("Starting batch processing...")
    extract_video_keypoints()
    print("\n[SUCCESS] All videos were converted into training-ready H5 files.")

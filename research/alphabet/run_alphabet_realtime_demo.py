"""Run an OpenCV smoke test for the static alphabet classifier."""

import cv2
import numpy as np
import mediapipe as mp
import collections
from tensorflow.keras.models import load_model
from alphabet_config import *

def draw_keypoints(image, results):
    """Draw hand landmarks for the alphabet OpenCV smoke-test preview."""
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def real_time_alphabet(threshold=0.80):
    """Run the static alphabet classifier against live webcam frames.

    A short voting buffer debounces predictions so the displayed letter changes only
    after several matching frames."""
    model = load_model(MODEL_PATH)
    mp_holistic = mp.solutions.holistic

    # Debounce predictions with a short rolling buffer.
    BUFFER_SIZE = 5
    predictions_buffer = collections.deque(maxlen=BUFFER_SIZE)
    current_letter = "-"

    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            image, results = mediapipe_detection(frame, holistic)
            draw_keypoints(image, results)

            if results.left_hand_landmarks or results.right_hand_landmarks:
                # Extract the current frame as a 306-value feature vector.
                keypoints = extract_keypoints(results)

                # Add the batch dimension expected by Keras: (1, 306).
                keypoints_reshaped = np.expand_dims(keypoints, axis=0)

                # Run one static-frame prediction.
                res = model.predict(keypoints_reshaped, verbose=0)[0]
                best_match_idx = np.argmax(res)

                if res[best_match_idx] > threshold:
                    predicted_char = ALPHABET[best_match_idx]
                else:
                    predicted_char = "-"

                predictions_buffer.append(predicted_char)

                # Vote across recent frames to stabilize the display.
                word_counts = collections.Counter(predictions_buffer)
                most_common, count = word_counts.most_common(1)[0]
                if count >= 3: # Require three of the last five frames to match.
                    current_letter = most_common
            else:
                current_letter = "-"
                predictions_buffer.clear()

            # --- OpenCV interface ---
            cv2.rectangle(image, (0, 0), (640, 60), (245, 117, 16), -1)
            cv2.putText(image, f"LETTER: {current_letter}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

            cv2.imshow('LESSA Alphabet Translator', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_alphabet(threshold=0.80)

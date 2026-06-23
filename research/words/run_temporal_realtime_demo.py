"""Run an OpenCV smoke test for the temporal word/phrase model.

This demo validates a trained artifact locally with webcam input. The thesis demo
path remains the FastAPI backend plus the Next.js frontend.
"""

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import collections
from word_config import *

# Keep the same drawing helper used by the research capture scripts.
def draw_keypoints(image, results):
    """Draw MediaPipe pose and hand landmarks for the OpenCV smoke-test preview."""
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def real_time_translation(threshold=0.65):
    """Run the active temporal word/phrase model against live webcam frames.

    The demo reproduces runtime inference: collect a rolling window, compute velocity
    and acceleration, pad to 60 frames, then stabilize predictions through voting."""
    model = load_model(MODEL_PATH)
    mp_holistic = mp.solutions.holistic

    WINDOW_SIZE = 25
    VOTING_BUFFER_SIZE = 10
    MIN_VOTES = 7

    sequence = collections.deque(maxlen=WINDOW_SIZE)
    predictions_buffer = collections.deque(maxlen=VOTING_BUFFER_SIZE)

    sentence = []
    last_emitted_word = "nada"
    rest_counter = 0
    current_probs = np.zeros(len(WORDS))

    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            image, results = mediapipe_detection(frame, holistic)
            draw_keypoints(image, results)

            # --- PHASE 1: POSITION EXTRACTION (306 values) ---
            if results.left_hand_landmarks or results.right_hand_landmarks:
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
            else:
                sequence.append(np.zeros(LENGTH_KEYPOINTS))

            # --- PHASE 2: REAL-TIME KINEMATICS AND PREDICTION ---
            if len(sequence) == WINDOW_SIZE:

                # Convert the rolling memory into a numeric matrix.
                seq_array = np.array(sequence)

                # Compute velocity from the live rolling window.
                delta = np.vstack([seq_array[0:1, :], np.diff(seq_array, axis=0)])

                # Compute acceleration (delta-delta) on the fly.
                delta_delta = np.vstack([delta[0:1, :], np.diff(delta, axis=0)])

                # Merge position + velocity + acceleration (306 + 306 + 306 = 918 features).
                combined_seq = np.concatenate([seq_array, delta, delta_delta], axis=-1)

                # Pad to MAX_FRAMES (60) so the live tensor matches the trained model shape.
                pad_seq = pad_sequences([combined_seq], maxlen=MAX_FRAMES, padding='post', dtype='float32')

                # 6. Prediction
                res = model.predict(pad_seq, verbose=0)[0]
                current_probs = res
                best_match_idx = np.argmax(res)

                if res[best_match_idx] > threshold:
                    current_word = WORDS[best_match_idx]
                else:
                    current_word = "nada"

                predictions_buffer.append(current_word)

                # --- PHASE 3: STATE MACHINE AND SMOOTHING ---
                word_counts = collections.Counter(predictions_buffer)
                most_common_word, count = word_counts.most_common(1)[0]

                if count >= MIN_VOTES:
                    stable_word = most_common_word
                else:
                    stable_word = "nada"

                if stable_word != "nada":
                    rest_counter = 0
                    if stable_word != last_emitted_word:
                        sentence.append(stable_word)
                        last_emitted_word = stable_word
                        if len(sentence) > 5:
                            sentence = sentence[-5:]
                else:
                    rest_counter += 1
                    if rest_counter > 15:
                        last_emitted_word = "nada"

            # --- GRAPHICAL INTERFACE ---
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence).upper(), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            panel_height = 60 + (len(WORDS) * 30)
            cv2.rectangle(image, (0, 50), (250, panel_height), (40, 40, 40), -1)

            y_offset = 80
            for i, word in enumerate(WORDS):
                prob = current_probs[i]
                color = (0, 255, 0) if (prob == np.max(current_probs) and prob > threshold) else (255, 255, 255)
                text = f"{word}: {prob*100:.1f}%"
                cv2.putText(image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
                y_offset += 30

            cv2.imshow('LESSA CSLR Translator', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_translation(threshold=0.65)

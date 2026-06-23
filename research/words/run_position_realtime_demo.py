"""Run an OpenCV smoke test for the position-only word/phrase experiment."""

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import collections
from word_config import *

def draw_keypoints(image, results):
    """Draw MediaPipe pose and hand landmarks for the OpenCV smoke-test preview."""
    # Use MediaPipe's standard drawing helper for the live interface.
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def real_time_translation(threshold=0.75):
    """Run the position-only word/phrase model against live webcam frames.

    This demo keeps a shorter live window and pads it to the trained 60-frame shape so
    latency stays acceptable while preserving the model input contract."""
    model = load_model(MODEL_PATH)
    mp_holistic = mp.solutions.holistic

    # 1. Continuous-sign parameters; tune these to match signer speed.
    WINDOW_SIZE = 40           # Sliding-window size, roughly one second of video.
    VOTING_BUFFER_SIZE = 15    # Recent predictions used for smoothing.
    MIN_VOTES = 10             # Votes required before confirming a word.

    # 2. Continuous rolling state.
    sequence = collections.deque(maxlen=WINDOW_SIZE)
    predictions_buffer = collections.deque(maxlen=VOTING_BUFFER_SIZE)

    # 3. State machine
    sentence = []
    last_emitted_word = "nada"
    rest_counter = 0           # Counts how long the model has remained in the rest state.
    current_probs = np.zeros(len(WORDS))

    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            image, results = mediapipe_detection(frame, holistic)
            draw_keypoints(image, results)

            # --- PHASE 1: EXTRACTION AND SLIDING WINDOW ---
            if results.left_hand_landmarks or results.right_hand_landmarks:
                # Reuse the shared config helper that normalizes landmarks relative to the nose.
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
            else:
                # Inject zeros when hands leave the frame so real-time flow remains stable.
                sequence.append(np.zeros(LENGTH_KEYPOINTS))

            # --- PHASE 2: CONTINUOUS PREDICTION ---
            if len(sequence) == WINDOW_SIZE:
                # The model expects MAX_FRAMES (60), so pad the shorter live window with zeros.
                # The Masking layer ignores this mathematical padding.
                pad_seq = pad_sequences([list(sequence)], maxlen=MAX_FRAMES, padding='post', dtype='float32')

                res = model.predict(pad_seq, verbose=0)[0]
                current_probs = res
                best_match_idx = np.argmax(res)

                if res[best_match_idx] > threshold:
                    current_word = WORDS[best_match_idx]
                else:
                    current_word = "nada"

                # Add the prediction to the voting buffer.
                predictions_buffer.append(current_word)

                # --- PHASE 3: SMOOTHING AND TRANSITION LOGIC ---
                # Confirm the most repeated word in the recent voting window.
                word_counts = collections.Counter(predictions_buffer)
                most_common_word, count = word_counts.most_common(1)[0]

                if count >= MIN_VOTES:
                    stable_word = most_common_word
                else:
                    stable_word = "nada"

                # State machine that appends confirmed words to the sentence.
                if stable_word != "nada":
                    rest_counter = 0 # Reset the rest-state counter.

                    # Only append a new word once to avoid repeated output such as 'hola hola hola'.
                    if stable_word != last_emitted_word:
                        sentence.append(stable_word)
                        last_emitted_word = stable_word

                        # Keep the sentence capped so the overlay remains readable.
                        if len(sentence) > 5:
                            sentence = sentence[-5:]

                else:
                    # Sustained rest lets the user repeat the last emitted word later.
                    rest_counter += 1
                    if rest_counter > 15: # Roughly half a second of true pause.
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
    real_time_translation(threshold=0.75)

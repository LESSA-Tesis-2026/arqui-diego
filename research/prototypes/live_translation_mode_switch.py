"""OpenCV prototype with manual switching between word and alphabet modes.

Useful for checking mode-transition behavior before changes are ported to the
FastAPI/WebSocket runtime.
"""

import cv2
from pathlib import Path
import sys

# Allow these standalone prototype scripts to be run from the repository root
# without installing the research package as a Python distribution.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import mediapipe as mp
import collections
import textwrap
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Import both research configuration modules with aliases to avoid name collisions.
import research.words.word_config as cfg_words
import research.alphabet.alphabet_config as cfg_alphabet

def draw_keypoints(image, results):
    """Draw pose and hand landmarks when debugging the manual hybrid prototype preview."""
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def real_time_combined(threshold_words=0.65, threshold_alphabet=0.80):
    """Run the manual hybrid prototype with keyboard switching between words and alphabet.

    Use this when isolating word or alphabet behavior without the automatic motion
    router influencing predictions."""
    print("Loading word model...")
    model_words = load_model(cfg_words.MODEL_PATH)
    print("Loading alphabet model...")
    model_alphabet = load_model(cfg_alphabet.MODEL_PATH)

    mp_holistic = mp.solutions.holistic

    # --- Hyperparameters and state ---
    current_mode = "words"

    # Accumulated translated text.
    accumulated_text = ""

    # Word-recognition state.
    WINDOW_SIZE = 25
    VOTING_BUFFER_SIZE_WORDS = 10
    MIN_VOTES_WORDS = 7
    sequence = collections.deque(maxlen=WINDOW_SIZE)
    predictions_buffer_words = collections.deque(maxlen=VOTING_BUFFER_SIZE_WORDS)
    last_emitted_word = "nada"
    rest_counter = 0
    current_probs = np.zeros(len(cfg_words.WORDS))

    # Alphabet-recognition state.
    ALPHABET_BUFFER_SIZE = 5
    alphabet_predictions_buffer = collections.deque(maxlen=ALPHABET_BUFFER_SIZE)
    last_emitted_letter = "-"  # Avoid infinitely repeating the same accepted letter.

    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            image, results = cfg_words.mediapipe_detection(frame, holistic)
            draw_keypoints(image, results)

            # ==========================================
            # MODE: word/phrase translation (CSLR).
            # ==========================================
            if current_mode == "words":
                if results.left_hand_landmarks or results.right_hand_landmarks:
                    keypoints = cfg_words.extract_keypoints(results)
                    sequence.append(keypoints)
                else:
                    sequence.append(np.zeros(cfg_words.LENGTH_KEYPOINTS))

                if len(sequence) == WINDOW_SIZE:
                    seq_array = np.array(sequence)
                    delta = np.vstack([seq_array[0:1, :], np.diff(seq_array, axis=0)])
                    delta_delta = np.vstack([delta[0:1, :], np.diff(delta, axis=0)])
                    combined_seq = np.concatenate([seq_array, delta, delta_delta], axis=-1)

                    pad_seq = pad_sequences([combined_seq], maxlen=cfg_words.MAX_FRAMES, padding='post', dtype='float32')
                    res = model_words.predict(pad_seq, verbose=0)[0]
                    current_probs = res
                    best_match_idx = np.argmax(res)

                    if res[best_match_idx] > threshold_words:
                        current_word = cfg_words.WORDS[best_match_idx]
                    else:
                        current_word = "nada"

                    predictions_buffer_words.append(current_word)

                    word_counts = collections.Counter(predictions_buffer_words)
                    most_common_word, count = word_counts.most_common(1)[0]

                    if count >= MIN_VOTES_WORDS: stable_word = most_common_word
                    else: stable_word = "nada"

                    if stable_word != "nada":
                        rest_counter = 0
                        if stable_word != last_emitted_word:
                            # Insert a space when appending a word after existing text.
                            if len(accumulated_text) > 0 and not accumulated_text.endswith(" "):
                                accumulated_text += " "
                            accumulated_text += stable_word + " "
                            last_emitted_word = stable_word
                    else:
                        rest_counter += 1
                        if rest_counter > 15:
                            last_emitted_word = "nada"

                # Word probability panel; placed below the persistent text area.
                panel_height = 110 + (len(cfg_words.WORDS) * 30)
                cv2.rectangle(image, (0, 110), (250, panel_height), (40, 40, 40), -1)

                y_offset = 140
                for i, word in enumerate(cfg_words.WORDS):
                    prob = current_probs[i]
                    color = (0, 255, 0) if (prob == np.max(current_probs) and prob > threshold_words) else (255, 255, 255)
                    text = f"{word}: {prob*100:.1f}%"
                    cv2.putText(image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
                    y_offset += 30

            # ==========================================
            # MODE: alphabet translation.
            # ==========================================
            elif current_mode == "alphabet":
                if results.left_hand_landmarks or results.right_hand_landmarks:
                    keypoints = cfg_alphabet.extract_keypoints(results)
                    keypoints_reshaped = np.expand_dims(keypoints, axis=0)

                    res = model_alphabet.predict(keypoints_reshaped, verbose=0)[0]
                    best_match_idx = np.argmax(res)

                    if res[best_match_idx] > threshold_alphabet:
                        predicted_char = cfg_alphabet.ALPHABET[best_match_idx]
                    else:
                        predicted_char = "-"

                    alphabet_predictions_buffer.append(predicted_char)

                    word_counts = collections.Counter(alphabet_predictions_buffer)
                    most_common, count = word_counts.most_common(1)[0]

                    if count >= 3:
                        current_letter = most_common
                    else:
                        current_letter = "-"

                    # Logic for joining consecutive letters (H + O + L + A).
                    if current_letter != "-" and current_letter != last_emitted_letter:
                        accumulated_text += current_letter
                        last_emitted_letter = current_letter
                    elif current_letter == "-":
                        # Reset duplicate-letter guard when the sign is no longer stable.
                        # This allows repeated letters, e.g. 'L', release, then 'L' again for 'LL'.
                        last_emitted_letter = "-"

                else:
                    last_emitted_letter = "-"
                    alphabet_predictions_buffer.clear()

            # ==========================================
            # Shared OpenCV interface with persistent accumulated text.
            # ==========================================

            # Text background, sized for multiple lines.
            cv2.rectangle(image, (0, 0), (640, 110), (245, 117, 16), -1)

            # Wrap text so it does not leave the frame, roughly 35 characters per line.
            wrapped_lines = textwrap.wrap(accumulated_text, width=35)

            # Render only the last three lines to avoid overflowing the overlay box.
            y_text = 35
            for line in wrapped_lines[-3:]:
                cv2.putText(image, line.upper(), (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                y_text += 35

            # --- General mode indicator ---
            mode_text = "MODE: WORDS" if current_mode == "words" else "MODE: ALPHABET"
            cv2.putText(image, f"{mode_text} ('s' = switch | 'b' = clear)", (10, image.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('LESSA Hybrid Translator', image)

            # --- Keyboard controls ---
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Switch mode.
                if current_mode == "words":
                    current_mode = "alphabet"
                    alphabet_predictions_buffer.clear()
                    last_emitted_letter = "-"
                    # Optionally add a space when switching into letter mode if one is missing.
                    if len(accumulated_text) > 0 and not accumulated_text.endswith(" "):
                        accumulated_text += " "
                else:
                    current_mode = "words"
                    sequence.clear()
                    predictions_buffer_words.clear()
                    last_emitted_word = "nada"
            elif key == ord('b'):
                # Clear all emitted text.
                accumulated_text = ""
                last_emitted_word = "nada"
                last_emitted_letter = "-"

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_combined()

"""OpenCV prototype for automatic switching between word and alphabet models.

Provides a research reference for hybrid-mode behavior outside the web/API stack.
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
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Import both research configuration modules with explicit model-path aliases.
from research.words.word_config import WORDS, MAX_FRAMES, MODEL_PATH as WORD_MODEL_PATH, LENGTH_KEYPOINTS as LENGTH_KEYPOINTS_WORDS, extract_keypoints as extract_keypoints_words, mediapipe_detection
from research.alphabet.alphabet_config import ALPHABET, MODEL_PATH as ALPHABET_MODEL_PATH, extract_keypoints as extract_keypoints_alphabet

def wrap_text_to_width(text, max_width, font, font_scale, thickness):
    """Wrap OpenCV overlay text by rendered pixel width instead of character count.

    Spanish phrases and joined letters have variable visual widths, so measuring with
    OpenCV prevents text from overflowing the overlay box."""
    if not text:
        return [""]

    tokens = text.split(" ")
    lines = []
    current = ""

    for token in tokens:
        candidate = token if current == "" else f"{current} {token}"
        if cv2.getTextSize(candidate, font, font_scale, thickness)[0][0] <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)
            current = ""

        for ch in token:
            candidate = ch if current == "" else f"{current}{ch}"
            if cv2.getTextSize(candidate, font, font_scale, thickness)[0][0] <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = ch

    if current:
        lines.append(current)

    return lines

def draw_keypoints(image, results):
    """Draw pose and hand landmarks when debugging the hybrid prototype preview."""
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def unified_real_time_translation(word_threshold=0.80, letter_threshold=0.85, motion_threshold=0.005):
    """Run the automatic hybrid prototype that routes motion to words and stillness to letters.

    This script documents the heuristic used before the web/API implementation: dynamic
    hand motion feeds the temporal word model, while static posture feeds the alphabet
    model."""
    print("Loading dynamic word LSTM model...")
    word_model = load_model(WORD_MODEL_PATH)

    print("Loading static alphabet dense model...")
    alphabet_model = load_model(ALPHABET_MODEL_PATH)
    print("Models loaded successfully. Starting camera...")

    mp_holistic = mp.solutions.holistic

    WINDOW_SIZE = 25
    sequence = collections.deque(maxlen=WINDOW_SIZE)

    word_buffer = collections.deque(maxlen=10)
    letter_buffer = collections.deque(maxlen=5)

    sentence = []
    last_emitted_item = ""
    rest_counter = 0

    # Stability counters prevent rapid switching between dynamic and static recognition.
    dynamic_frames = 0
    static_frames = 0

    current_status = "SEARCHING FOR HANDS"
    current_word = "-"
    current_letter = "-"

    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            image, results = mediapipe_detection(frame, holistic)
            #draw_keypoints(image, results)

            hands_detected = bool(results.left_hand_landmarks or results.right_hand_landmarks)

            if hands_detected:
                keypoints_words = extract_keypoints_words(results)
                keypoints_letters = extract_keypoints_alphabet(results)
                sequence.append(keypoints_words)
                rest_counter = 0
            else:
                keypoints_letters = None
                sequence.append(np.zeros(LENGTH_KEYPOINTS_WORDS))
                current_status = "NO HANDS"
                word_buffer.clear()
                letter_buffer.clear()
                rest_counter += 1

            # --- Motion-based routing between word and alphabet models ---
            if len(sequence) == WINDOW_SIZE and hands_detected:
                seq_array = np.array(sequence)
                delta = np.vstack([seq_array[0:1, :], np.diff(seq_array, axis=0)])

                # Average the last five frames to reduce hand-tremor noise in the motion router.
                hand_motion_speed = np.mean(np.abs(delta[-5:, 180:]))

                # --- Word mode (dynamic movement) ---
                if hand_motion_speed > motion_threshold:
                    dynamic_frames += 1
                    static_frames = 0
                    current_status = f"WORD (speed: {hand_motion_speed:.4f})"

                    # Sustained motion means we are transitioning to a word, so clear partial letters.
                    if dynamic_frames > 4:
                        letter_buffer.clear()
                        # Allow repeated letters after a clear hand movement separates emissions.
                        if len(last_emitted_item) == 1:
                            last_emitted_item = ""

                    # Run the LSTM with position + delta + delta-delta features.
                    delta_delta = np.vstack([delta[0:1, :], np.diff(delta, axis=0)])
                    combined_seq = np.concatenate([seq_array, delta, delta_delta], axis=-1)
                    pad_seq = pad_sequences([combined_seq], maxlen=MAX_FRAMES, padding='post', dtype='float32')

                    res_word = word_model.predict(pad_seq, verbose=0)[0]
                    best_idx = np.argmax(res_word)

                    if res_word[best_idx] > word_threshold:
                        predicted_word = WORDS[best_idx]
                        current_word = f"{predicted_word} ({res_word[best_idx]*100:.0f}%)"
                    else:
                        predicted_word = "nada"
                        current_word = "-"

                    word_buffer.append(predicted_word)

                # --- Alphabet mode (static posture) ---
                else:
                    static_frames += 1
                    dynamic_frames = 0
                    current_status = f"LETTER (speed: {hand_motion_speed:.4f})"

                    # Sustained stillness means partial word votes should be cleared.
                    if static_frames > 4:
                        word_buffer.clear()

                    # Run the dense alphabet classifier.
                    if keypoints_letters is not None:
                        latest_frame = np.expand_dims(keypoints_letters, axis=0)
                        res_letter = alphabet_model.predict(latest_frame, verbose=0)[0]
                        best_idx = np.argmax(res_letter)

                        if res_letter[best_idx] > letter_threshold:
                            letter_buffer.append(ALPHABET[best_idx])
                            current_letter = f"{ALPHABET[best_idx]} ({res_letter[best_idx]*100:.0f}%)"
                        else:
                            letter_buffer.append("-")
                            current_letter = "-"
                    else:
                        letter_buffer.clear()
                        current_letter = "-"

            # --- Strict emission state machine ---
            # 1. Render word probabilities, matching the original prototype layout.
            if len(word_buffer) > 0:
                counts = collections.Counter(word_buffer)
                most_common, votes = counts.most_common(1)[0]
                if votes >= 7 and most_common != "nada" and most_common != last_emitted_item:
                    sentence.append(f" {most_common} ")
                    last_emitted_item = most_common

            # 2. Render letters quickly and responsively.
            if len(letter_buffer) > 0:
                counts = collections.Counter(letter_buffer)
                most_common, votes = counts.most_common(1)[0]
                if votes >= 3 and most_common != "-" and most_common != last_emitted_item:
                    sentence.append(most_common)
                    last_emitted_item = most_common

            # Reset duplicate-emission guard after a long no-hands pause.
            if rest_counter > 20:
                last_emitted_item = ""

            # --- OpenCV interface ---
            box_x0, box_y0 = 0, 0
            box_x1, box_y1 = 640, 120
            cv2.rectangle(image, (box_x0, box_y0), (box_x1, box_y1), (245, 117, 16), -1)

            display_text = "".join(sentence)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.9
            thickness = 2
            line_height = cv2.getTextSize("Ag", font, font_scale, thickness)[0][1] + 8
            max_width = box_x1 - box_x0 - 20
            max_lines = max(1, (box_y1 - box_y0 - 10) // line_height)

            lines = wrap_text_to_width(display_text.upper(), max_width, font, font_scale, thickness)
            lines = lines[-max_lines:]

            y = box_y0 + 30
            for line in lines:
                cv2.putText(image, line, (10, y), font, font_scale, (255, 255, 255), thickness)
                y += line_height

            cv2.rectangle(image, (0, 130), (320, 250), (40, 40, 40), -1)
            status_color = (0, 255, 0) if "LETTER" in current_status else (0, 165, 255)
            if "SEARCHING" in current_status or "NO" in current_status: status_color = (150, 150, 150)

            cv2.putText(image, f"MODE: {current_status}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            cv2.putText(image, f"Word pred.: {current_word}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(image, f"Letter pred.: {current_letter}", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('LESSA Hybrid Translator', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Thresholds tuned for maximum stability.
    unified_real_time_translation(word_threshold=0.70, letter_threshold=0.80, motion_threshold=0.001)

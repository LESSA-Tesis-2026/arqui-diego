"""Prototipo de OpenCV para el cambio automático entre los modelos de palabras y de alfabeto.

Provee una referencia de investigación para el comportamiento del modo híbrido fuera del stack web/API.
"""

import cv2
from pathlib import Path
import sys

# Permite que estos scripts de prototipo independientes se ejecuten desde la raíz del repositorio
# sin instalar el paquete de research como una distribución de Python.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import mediapipe as mp
import collections
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Importa ambos módulos de configuración de research con alias explícitos de ruta de modelo.
from research.words.word_config import WORDS, MAX_FRAMES, MODEL_PATH as WORD_MODEL_PATH, LENGTH_KEYPOINTS as LENGTH_KEYPOINTS_WORDS, extract_keypoints as extract_keypoints_words, mediapipe_detection
from research.alphabet.alphabet_config import ALPHABET, MODEL_PATH as ALPHABET_MODEL_PATH, extract_keypoints as extract_keypoints_alphabet

def wrap_text_to_width(text, max_width, font, font_scale, thickness):
    """Ajusta el texto de la superposición de OpenCV por ancho de píxel renderizado en lugar de por conteo de caracteres.

    Las frases en español y las letras unidas tienen anchos visuales variables, así que medir con
    OpenCV evita que el texto se desborde de la caja de superposición."""
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
    """Dibuja los landmarks de pose y manos al depurar la vista previa del prototipo híbrido."""
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def unified_real_time_translation(word_threshold=0.80, letter_threshold=0.85, motion_threshold=0.005):
    """Ejecuta el prototipo híbrido automático que enruta el movimiento a palabras y la quietud a letras.

    Este script documenta la heurística usada antes de la implementación web/API: el movimiento
    dinámico de la mano alimenta el modelo temporal de palabras, mientras que la postura estática alimenta el
    modelo del alfabeto."""
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

    # Los contadores de estabilidad evitan el cambio rápido entre el reconocimiento dinámico y el estático.
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

            # --- Enrutamiento basado en movimiento entre los modelos de palabras y de alfabeto ---
            if len(sequence) == WINDOW_SIZE and hands_detected:
                seq_array = np.array(sequence)
                delta = np.vstack([seq_array[0:1, :], np.diff(seq_array, axis=0)])

                # Promedia los últimos cinco fotogramas para reducir el ruido del temblor de la mano en el enrutador de movimiento.
                hand_motion_speed = np.mean(np.abs(delta[-5:, 180:]))

                # --- Modo palabra (movimiento dinámico) ---
                if hand_motion_speed > motion_threshold:
                    dynamic_frames += 1
                    static_frames = 0
                    current_status = f"WORD (speed: {hand_motion_speed:.4f})"

                    # Un movimiento sostenido significa que estamos transicionando a una palabra, así que limpia las letras parciales.
                    if dynamic_frames > 4:
                        letter_buffer.clear()
                        # Permite letras repetidas después de que un movimiento claro de la mano separe las emisiones.
                        if len(last_emitted_item) == 1:
                            last_emitted_item = ""

                    # Ejecuta la LSTM con las características de posición + delta + delta-delta.
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

                # --- Modo alfabeto (postura estática) ---
                else:
                    static_frames += 1
                    dynamic_frames = 0
                    current_status = f"LETTER (speed: {hand_motion_speed:.4f})"

                    # Una quietud sostenida significa que se deben limpiar los votos parciales de palabra.
                    if static_frames > 4:
                        word_buffer.clear()

                    # Ejecuta el clasificador denso del alfabeto.
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

            # --- Máquina de estados de emisión estricta ---
            # 1. Renderiza las probabilidades de palabra, replicando la disposición del prototipo original.
            if len(word_buffer) > 0:
                counts = collections.Counter(word_buffer)
                most_common, votes = counts.most_common(1)[0]
                if votes >= 7 and most_common != "nada" and most_common != last_emitted_item:
                    sentence.append(f" {most_common} ")
                    last_emitted_item = most_common

            # 2. Renderiza las letras de forma rápida y con buena respuesta.
            if len(letter_buffer) > 0:
                counts = collections.Counter(letter_buffer)
                most_common, votes = counts.most_common(1)[0]
                if votes >= 3 and most_common != "-" and most_common != last_emitted_item:
                    sentence.append(most_common)
                    last_emitted_item = most_common

            # Reinicia la protección contra emisiones duplicadas tras una larga pausa sin manos.
            if rest_counter > 20:
                last_emitted_item = ""

            # --- Interfaz de OpenCV ---
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
    # Umbrales ajustados para máxima estabilidad.
    unified_real_time_translation(word_threshold=0.70, letter_threshold=0.80, motion_threshold=0.001)

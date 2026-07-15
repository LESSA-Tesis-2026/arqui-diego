"""Prototipo de OpenCV con cambio manual entre los modos de palabras y de alfabeto.

Útil para verificar el comportamiento de transición entre modos antes de portar los cambios al
entorno de ejecución de FastAPI/WebSocket.
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
import textwrap
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Importa ambos módulos de configuración de research con alias para evitar colisiones de nombres.
import research.words.word_config as cfg_words
import research.alphabet.alphabet_config as cfg_alphabet

def draw_keypoints(image, results):
    """Dibuja los landmarks de pose y manos al depurar la vista previa del prototipo híbrido manual."""
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def real_time_combined(threshold_words=0.65, threshold_alphabet=0.80):
    """Ejecuta el prototipo híbrido manual con cambio por teclado entre palabras y alfabeto.

    Úsalo cuando aísles el comportamiento de palabras o de alfabeto sin que el enrutador
    automático de movimiento influya en las predicciones."""
    print("Loading word model...")
    model_words = load_model(cfg_words.MODEL_PATH)
    print("Loading alphabet model...")
    model_alphabet = load_model(cfg_alphabet.MODEL_PATH)

    mp_holistic = mp.solutions.holistic

    # --- Hiperparámetros y estado ---
    current_mode = "words"

    # Texto traducido acumulado.
    accumulated_text = ""

    # Estado del reconocimiento de palabras.
    WINDOW_SIZE = 25
    VOTING_BUFFER_SIZE_WORDS = 10
    MIN_VOTES_WORDS = 7
    sequence = collections.deque(maxlen=WINDOW_SIZE)
    predictions_buffer_words = collections.deque(maxlen=VOTING_BUFFER_SIZE_WORDS)
    last_emitted_word = "nada"
    rest_counter = 0
    current_probs = np.zeros(len(cfg_words.WORDS))

    # Estado del reconocimiento del alfabeto.
    ALPHABET_BUFFER_SIZE = 5
    alphabet_predictions_buffer = collections.deque(maxlen=ALPHABET_BUFFER_SIZE)
    last_emitted_letter = "-"  # Evita repetir infinitamente la misma letra aceptada.

    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            image, results = cfg_words.mediapipe_detection(frame, holistic)
            draw_keypoints(image, results)

            # ==========================================
            # MODO: traducción de palabras/frases (CSLR).
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
                            # Inserta un espacio al agregar una palabra después del texto existente.
                            if len(accumulated_text) > 0 and not accumulated_text.endswith(" "):
                                accumulated_text += " "
                            accumulated_text += stable_word + " "
                            last_emitted_word = stable_word
                    else:
                        rest_counter += 1
                        if rest_counter > 15:
                            last_emitted_word = "nada"

                # Panel de probabilidades de palabra; ubicado debajo del área de texto persistente.
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
            # MODO: traducción del alfabeto.
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

                    # Lógica para unir letras consecutivas (H + O + L + A).
                    if current_letter != "-" and current_letter != last_emitted_letter:
                        accumulated_text += current_letter
                        last_emitted_letter = current_letter
                    elif current_letter == "-":
                        # Reinicia la protección contra letras duplicadas cuando la seña deja de ser estable.
                        # Esto permite letras repetidas, p. ej. 'L', soltar y luego 'L' de nuevo para 'LL'.
                        last_emitted_letter = "-"

                else:
                    last_emitted_letter = "-"
                    alphabet_predictions_buffer.clear()

            # ==========================================
            # Interfaz compartida de OpenCV con texto acumulado persistente.
            # ==========================================

            # Fondo del texto, dimensionado para múltiples líneas.
            cv2.rectangle(image, (0, 0), (640, 110), (245, 117, 16), -1)

            # Ajusta el texto para que no salga del fotograma, aproximadamente 35 caracteres por línea.
            wrapped_lines = textwrap.wrap(accumulated_text, width=35)

            # Renderiza solo las últimas tres líneas para evitar desbordar la caja de superposición.
            y_text = 35
            for line in wrapped_lines[-3:]:
                cv2.putText(image, line.upper(), (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                y_text += 35

            # --- Indicador general de modo ---
            mode_text = "MODE: WORDS" if current_mode == "words" else "MODE: ALPHABET"
            cv2.putText(image, f"{mode_text} ('s' = switch | 'b' = clear)", (10, image.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('LESSA Hybrid Translator', image)

            # --- Controles de teclado ---
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Cambia de modo.
                if current_mode == "words":
                    current_mode = "alphabet"
                    alphabet_predictions_buffer.clear()
                    last_emitted_letter = "-"
                    # Opcionalmente agrega un espacio al cambiar al modo de letras si falta uno.
                    if len(accumulated_text) > 0 and not accumulated_text.endswith(" "):
                        accumulated_text += " "
                else:
                    current_mode = "words"
                    sequence.clear()
                    predictions_buffer_words.clear()
                    last_emitted_word = "nada"
            elif key == ord('b'):
                # Limpia todo el texto emitido.
                accumulated_text = ""
                last_emitted_word = "nada"
                last_emitted_letter = "-"

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_combined()

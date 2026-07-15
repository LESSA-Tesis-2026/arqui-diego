"""Ejecuta una prueba rápida (smoke test) de OpenCV para el modelo temporal de palabras/frases.

Este demo valida un artefacto entrenado localmente con entrada de webcam. La ruta de la demo de
tesis sigue siendo el backend de FastAPI más el frontend de Next.js.
"""

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import collections
from word_config import *

# Conserva el mismo ayudante de dibujo usado por los scripts de captura de investigación.
def draw_keypoints(image, results):
    """Dibuja los landmarks de pose y manos de MediaPipe para la vista previa de prueba rápida (smoke-test) de OpenCV."""
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def real_time_translation(threshold=0.65):
    """Ejecuta el modelo temporal activo de palabras/frases sobre fotogramas de webcam en vivo.

    El demo reproduce la inferencia de ejecución: recolecta una ventana deslizante, calcula la velocidad
    y la aceleración, rellena a 60 fotogramas y luego estabiliza las predicciones mediante votación."""
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

            # --- FASE 1: EXTRACCIÓN DE POSICIÓN (306 valores) ---
            if results.left_hand_landmarks or results.right_hand_landmarks:
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
            else:
                sequence.append(np.zeros(LENGTH_KEYPOINTS))

            # --- FASE 2: CINEMÁTICA Y PREDICCIÓN EN TIEMPO REAL ---
            if len(sequence) == WINDOW_SIZE:

                # Convierte la memoria deslizante en una matriz numérica.
                seq_array = np.array(sequence)

                # Calcula la velocidad a partir de la ventana deslizante en vivo.
                delta = np.vstack([seq_array[0:1, :], np.diff(seq_array, axis=0)])

                # Calcula la aceleración (delta-delta) sobre la marcha.
                delta_delta = np.vstack([delta[0:1, :], np.diff(delta, axis=0)])

                # Combina posición + velocidad + aceleración (306 + 306 + 306 = 918 características).
                combined_seq = np.concatenate([seq_array, delta, delta_delta], axis=-1)

                # Rellena hasta MAX_FRAMES (60) para que el tensor en vivo coincida con la forma del modelo entrenado.
                pad_seq = pad_sequences([combined_seq], maxlen=MAX_FRAMES, padding='post', dtype='float32')

                # 6. Predicción
                res = model.predict(pad_seq, verbose=0)[0]
                current_probs = res
                best_match_idx = np.argmax(res)

                if res[best_match_idx] > threshold:
                    current_word = WORDS[best_match_idx]
                else:
                    current_word = "nada"

                predictions_buffer.append(current_word)

                # --- FASE 3: MÁQUINA DE ESTADOS Y SUAVIZADO ---
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

            # --- INTERFAZ GRÁFICA ---
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

"""Ejecuta una prueba rápida (smoke test) de OpenCV para el experimento de palabras/frases solo de posición."""

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import collections
from word_config import *

def draw_keypoints(image, results):
    """Dibuja los landmarks de pose y manos de MediaPipe para la vista previa de prueba rápida (smoke-test) de OpenCV."""
    # Usa el ayudante de dibujo estándar de MediaPipe para la interfaz en vivo.
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def real_time_translation(threshold=0.75):
    """Ejecuta el modelo de palabras/frases solo de posición sobre fotogramas de webcam en vivo.

    Este demo mantiene una ventana en vivo más corta y la rellena hasta la forma entrenada de 60 fotogramas para que
    la latencia se mantenga aceptable preservando el contrato de entrada del modelo."""
    model = load_model(MODEL_PATH)
    mp_holistic = mp.solutions.holistic

    # 1. Parámetros de seña continua; ajústalos para que coincidan con la velocidad del señante.
    WINDOW_SIZE = 40           # Tamaño de la ventana deslizante, aproximadamente un segundo de video.
    VOTING_BUFFER_SIZE = 15    # Predicciones recientes usadas para el suavizado.
    MIN_VOTES = 10             # Votos requeridos antes de confirmar una palabra.

    # 2. Estado deslizante continuo.
    sequence = collections.deque(maxlen=WINDOW_SIZE)
    predictions_buffer = collections.deque(maxlen=VOTING_BUFFER_SIZE)

    # 3. Máquina de estados
    sentence = []
    last_emitted_word = "nada"
    rest_counter = 0           # Cuenta cuánto tiempo ha permanecido el modelo en el estado de reposo.
    current_probs = np.zeros(len(WORDS))

    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            image, results = mediapipe_detection(frame, holistic)
            draw_keypoints(image, results)

            # --- FASE 1: EXTRACCIÓN Y VENTANA DESLIZANTE ---
            if results.left_hand_landmarks or results.right_hand_landmarks:
                # Reutiliza el ayudante de configuración compartido que normaliza los landmarks con respecto a la nariz.
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
            else:
                # Inyecta ceros cuando las manos salen del fotograma para que el flujo en tiempo real se mantenga estable.
                sequence.append(np.zeros(LENGTH_KEYPOINTS))

            # --- FASE 2: PREDICCIÓN CONTINUA ---
            if len(sequence) == WINDOW_SIZE:
                # El modelo espera MAX_FRAMES (60), así que rellena con ceros la ventana en vivo más corta.
                # La capa de Masking ignora este relleno matemático.
                pad_seq = pad_sequences([list(sequence)], maxlen=MAX_FRAMES, padding='post', dtype='float32')

                res = model.predict(pad_seq, verbose=0)[0]
                current_probs = res
                best_match_idx = np.argmax(res)

                if res[best_match_idx] > threshold:
                    current_word = WORDS[best_match_idx]
                else:
                    current_word = "nada"

                # Agrega la predicción al búfer de votación.
                predictions_buffer.append(current_word)

                # --- FASE 3: LÓGICA DE SUAVIZADO Y TRANSICIÓN ---
                # Confirma la palabra más repetida en la ventana de votación reciente.
                word_counts = collections.Counter(predictions_buffer)
                most_common_word, count = word_counts.most_common(1)[0]

                if count >= MIN_VOTES:
                    stable_word = most_common_word
                else:
                    stable_word = "nada"

                # Máquina de estados que agrega las palabras confirmadas a la oración.
                if stable_word != "nada":
                    rest_counter = 0 # Reinicia el contador del estado de reposo.

                    # Agrega una palabra nueva solo una vez para evitar salidas repetidas como 'hola hola hola'.
                    if stable_word != last_emitted_word:
                        sentence.append(stable_word)
                        last_emitted_word = stable_word

                        # Mantén la oración limitada para que la superposición siga siendo legible.
                        if len(sentence) > 5:
                            sentence = sentence[-5:]

                else:
                    # Un reposo sostenido permite que el usuario repita más tarde la última palabra emitida.
                    rest_counter += 1
                    if rest_counter > 15: # Aproximadamente medio segundo de pausa real.
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
    real_time_translation(threshold=0.75)

"""Recolecta muestras dinámicas de palabras/frases de LESSA directamente en archivos de secuencias H5.

Use este script cuando no se necesite revisar el video sin procesar. Agrega nuevas muestras al
archivo H5 existente por etiqueta, lo que permite reanudar una sesión de captura sin eliminar
las secuencias grabadas previamente.
"""

import cv2
import numpy as np
import h5py
import mediapipe as mp
import os
from word_config import *

def draw_custom_keypoints(image, results):
    """Dibuja la misma superposición compacta de landmarks que usan las herramientas de captura de palabras/frases.

    Solo se dibujan los puntos de rostro seleccionados porque la malla facial completa oculta las manos y
    dificulta revisar si una captura es utilizable."""
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    # Dibuja solo los landmarks de rostro seleccionados para mantener legible la superposición de captura.
    if results.face_landmarks:
        h, w, _ = image.shape
        for idx in SELECTED_FACE_INDICES:
            landmark = results.face_landmarks.landmark[idx]
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (cx, cy), 3, (0, 255, 255), -1)

    # Dibuja los landmarks de pose y manos con las conexiones estándar de MediaPipe.
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

def collect_dynamic_h5_sequences(word, target_samples=60):
    """Agrega nuevas secuencias de señas dinámicas de una etiqueta a su dataset H5.

    Cada muestra almacena una lista de longitud variable de vectores de fotograma de 306 valores. El entrenamiento
    rellena/trunca después, así que el paso de captura preserva la duración natural de la seña."""

    # La clase de reposo necesita ejemplos adicionales porque actúa como el estado de transición/sin salida.
    if word.lower() == "nada":
        target_samples = int(target_samples * 1.75)
        print(f"\n[INFO] Class 'nada' detected. Target adjusted automatically a {target_samples} samples.")

    create_folder_if_not_exists(DATA_PATH)
    file_path = os.path.join(DATA_PATH, f"{word}.h5")

    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # El modo de anexado (append) permite reanudar sesiones interrumpidas sin perder las muestras capturadas previamente.
        with h5py.File(file_path, 'a') as hf:

            existing_samples = len(hf.keys())

            # Omite las etiquetas que ya alcanzaron la cantidad de muestras objetivo solicitada.
            if existing_samples >= target_samples:
                print(f"[*] Word '{word.upper()}' already has {existing_samples} samples. Skipping...")
                cap.release()
                cv2.destroyAllWindows()
                return

            # Calcula solo las muestras faltantes para que el archivo H5 pueda crecer de forma incremental.
            samples_to_record = target_samples - existing_samples
            print(f"\n--- COLLECTING SAMPLES FOR: {word.upper()} ---")
            print(f"Existing samples: {existing_samples} | Remaining: {samples_to_record} to reach target of {target_samples}.")

            for i in range(samples_to_record):
                # Usa el índice absoluto de la muestra para evitar sobrescribir datasets existentes.
                current_sample_idx = existing_samples + i
                sequence_data = []
                recording = False

                while True:
                    ret, frame = cap.read()
                    if not ret: break

                    image, results = mediapipe_detection(frame, holistic)
                    draw_custom_keypoints(image, results)

                    # Muestra el progreso respecto a la cantidad total de muestras solicitada.
                    cv2.putText(image, f"Label: {word} | Sample: {current_sample_idx + 1}/{target_samples}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    if recording:
                        cv2.putText(image, "RECORDING... (press 's' to stop)", (10, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        keypoints = extract_keypoints(results)
                        sequence_data.append(keypoints)
                    else:
                        cv2.putText(image, "Press 'r' to start capture", (10, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    cv2.imshow('Data Capture', image)
                    key = cv2.waitKey(10) & 0xFF

                    if key == ord('r') and not recording:
                        recording = True
                    elif key == ord('s') and recording:
                        recording = False

                        # Almacena la secuencia con el índice absoluto de muestra estable.
                        dataset_name = f"sample_{current_sample_idx}"
                        hf.create_dataset(dataset_name, data=np.array(sequence_data))
                        print(f"Sample {current_sample_idx + 1} saved with {len(sequence_data)} frames.")
                        break
                    elif key == ord('q'):
                        print("\nRecording interrupted by the user.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Objetivo global: se conservan las muestras existentes y solo se agregan las faltantes.
    # ACTUAL: 80
    TARGET_SAMPLES = 80

    for word in WORDS:
        collect_dynamic_h5_sequences(word, target_samples=TARGET_SAMPLES)

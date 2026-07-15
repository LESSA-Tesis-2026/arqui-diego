"""Graba videos de webcam sin procesar para cada etiqueta de palabra/frase de LESSA.

Use este script cuando las muestras deban revisarse visualmente antes de la extracción de
keypoints. Ejecute `extract_video_keypoints.py` después para construir los datasets H5.
"""

import cv2
import os
import mediapipe as mp
from word_config import *


def draw_custom_keypoints(image, results):
    """Dibuja una superposición ligera para retroalimentación del operador mientras se guardan fotogramas limpios sin procesar."""
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    if results.face_landmarks:
        h, w, _ = image.shape
        for idx in SELECTED_FACE_INDICES:
            landmark = results.face_landmarks.landmark[idx]
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (cx, cy), 3, (0, 255, 255), -1)

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

def create_video_folders():
    """Asegura que cada etiqueta de palabra/frase configurada tenga una carpeta de destino para archivos AVI."""
    if not os.path.exists(VIDEOS_FOLDER):
        os.makedirs(VIDEOS_FOLDER)
    for word in WORDS:
        word_path = os.path.join(VIDEOS_FOLDER, word)
        if not os.path.exists(word_path):
            os.makedirs(word_path)

def collect_raw_videos(word, target_samples=80):
    """Graba muestras AVI sin procesar para una etiqueta sin incrustar la superposición de landmarks.

    La ventana de vista previa se anota para el operador, pero el archivo guardado recibe el
    fotograma original de la cámara para que la posterior extracción de keypoints vea una entrada realista."""
    if word.lower() == "nada":
        target_samples = int(target_samples * 1.75)
        print(f"\n[INFO] Class 'nada' detected. Target adjusted a {target_samples} videos.")

    word_folder = os.path.join(VIDEOS_FOLDER, word)

    # Cuenta los videos AVI existentes para que las sesiones de captura interrumpidas puedan reanudarse de forma segura.
    existing_videos = len([f for f in os.listdir(word_folder) if f.endswith('.avi')])

    if existing_videos >= target_samples:
        print(f"[*] The label '{word.upper()}' already has {existing_videos} videos. Skipping...")
        return

    samples_to_record = target_samples - existing_videos
    print(f"\n--- COLLECTING VIDEOS FOR: {word.upper()} ---")
    print(f"Existing videos: {existing_videos} | Remaining: {samples_to_record}")

    cap = cv2.VideoCapture(0)
    # Lee la resolución de la cámara para que los videos guardados coincidan con el tamaño del fotograma sin procesar.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30.0 # Fotogramas por segundo estándar.
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # Códec compatible y eficiente.

    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for i in range(samples_to_record):
            current_sample_idx = existing_videos + i
            recording = False
            out = None

            while True:
                ret, frame = cap.read()
                if not ret: break

                # Procesa el fotograma solo para retroalimentación visual en vivo.
                image, results = mediapipe_detection(frame, holistic)
                draw_custom_keypoints(image, results)

                cv2.putText(image, f"Label: {word} | Video: {current_sample_idx + 1}/{target_samples}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                if recording:
                    cv2.putText(image, "RECORDING... (press 's' to stop)", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # Importante: guarda el fotograma sin procesar, no la imagen de vista previa anotada.
                    out.write(frame)
                else:
                    cv2.putText(image, "Press 'r' to start capture", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('LESSA Video Capture', image)
                key = cv2.waitKey(10) & 0xFF

                if key == ord('r') and not recording:
                    recording = True
                    # Inicializa el escritor de video cuando comienza la grabación.
                    video_path = os.path.join(word_folder, f"sample_{current_sample_idx}.avi")
                    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

                elif key == ord('s') and recording:
                    recording = False
                    out.release() # Cierra y persiste el archivo de video.
                    print(f"Video {current_sample_idx + 1} saved successfully.")
                    break

                elif key == ord('q'):
                    print("\nRecording interrupted by the user.")
                    if out is not None: out.release()
                    cap.release()
                    cv2.destroyAllWindows()
                    return

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    TARGET_SAMPLES = 80
    create_video_folders()
    for word in WORDS:
        collect_raw_videos(word, target_samples=TARGET_SAMPLES)

"""Convierte videos de palabras/frases sin procesar en datasets H5 de keypoints de MediaPipe.

El extractor reutiliza `word_config.extract_keypoints` para que el dataset de investigación
coincida con el mismo contrato de características relativo a la nariz que esperan el entrenamiento y la ejecución.
"""

import cv2
import numpy as np
import h5py
import mediapipe as mp
import os
from word_config import *


def extract_video_keypoints():
    """Convierte por lotes videos de palabras/frases sin procesar en secuencias de keypoints H5 por etiqueta.

    Las claves de dataset existentes se omiten, lo que permite ejecutar el extractor varias
    veces tras sesiones de captura interrumpidas sin duplicar muestras."""
    # Asegura que la carpeta de salida exista antes de escribir los datasets H5.
    create_folder_if_not_exists(DATA_PATH)
    mp_holistic = mp.solutions.holistic

    # Inicia MediaPipe una sola vez y reutilízalo en todos los videos para un procesamiento por lotes más rápido.
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        for word in WORDS:
            word_folder = os.path.join(VIDEOS_FOLDER, word)
            if not os.path.exists(word_folder):
                print(f"[*] The video folder for '{word}' does not exist. Skipping...")
                continue

            h5_file_path = os.path.join(DATA_PATH, f"{word}.h5")
            video_files = [f for f in os.listdir(word_folder) if f.endswith('.avi')]

            if not video_files:
                continue

            print(f"\n--- EXTRACTING VIDEO LANDMARKS: {word.upper()} ---")

            # Abre el archivo H5 en modo de anexado (append) para que el procesamiento pueda reanudarse sin sobrescribir las muestras existentes.
            with h5py.File(h5_file_path, 'a') as hf:
                existing_datasets = list(hf.keys())

                for video_file in video_files:
                    # Convierte 'sample_0.avi' en la clave de dataset 'sample_0'.
                    dataset_name = os.path.splitext(video_file)[0]

                    # Omite los videos que ya se convirtieron en datasets H5.
                    if dataset_name in existing_datasets:
                        continue

                    video_path = os.path.join(word_folder, video_file)
                    cap = cv2.VideoCapture(video_path)
                    sequence_data = []

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break # Fin del video.

                        # Procesa el fotograma sin procesar a través de MediaPipe.
                        image, results = mediapipe_detection(frame, holistic)

                        # Usa el ayudante de configuración compartido, que normaliza las coordenadas con respecto a la nariz.
                        keypoints = extract_keypoints(results)
                        sequence_data.append(keypoints)

                    cap.release()

                    # Persiste la secuencia de keypoints extraída en el archivo H5.
                    if sequence_data:
                        hf.create_dataset(dataset_name, data=np.array(sequence_data))
                        print(f"Processed: {dataset_name} | Length: {len(sequence_data)} frames")
                    else:
                        print(f"WARNING: No keypoint information was detected in {video_file}")

if __name__ == "__main__":
    print("Starting batch processing...")
    extract_video_keypoints()
    print("\n[SUCCESS] All videos were converted into training-ready H5 files.")

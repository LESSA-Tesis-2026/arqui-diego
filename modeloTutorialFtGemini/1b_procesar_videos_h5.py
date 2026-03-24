import cv2
import numpy as np
import h5py
import mediapipe as mp
import os
from config import *


def process_videos_to_h5():
    # Asegurarnos de que exista la carpeta data/ (donde van los .h5)
    create_folder_if_not_exists(DATA_PATH)
    mp_holistic = mp.solutions.holistic

    # Iniciamos MediaPipe una sola vez para procesar todos los videos de forma óptima
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        for word in WORDS:
            word_folder = os.path.join(VIDEOS_FOLDER, word)
            if not os.path.exists(word_folder):
                print(f"[*] La carpeta de videos para '{word}' no existe. Saltando...")
                continue

            h5_file_path = os.path.join(DATA_PATH, f"{word}.h5")
            video_files = [f for f in os.listdir(word_folder) if f.endswith('.avi')]

            if not video_files:
                continue

            print(f"\n--- EXTRAYENDO LANDMARKS DE VIDEOS: {word.upper()} ---")

            # Abrimos el .h5 en modo append para añadir o leer lo existente
            with h5py.File(h5_file_path, 'a') as hf:
                existing_datasets = list(hf.keys())

                for video_file in video_files:
                    # Obtenemos el nombre base (ej. convertimos 'sample_0.avi' a 'sample_0')
                    dataset_name = os.path.splitext(video_file)[0]

                    # Si este video ya está en el .h5, lo saltamos
                    if dataset_name in existing_datasets:
                        continue

                    video_path = os.path.join(word_folder, video_file)
                    cap = cv2.VideoCapture(video_path)
                    sequence_data = []

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break # Fin del video

                        # Pasamos el frame crudo por MediaPipe
                        image, results = mediapipe_detection(frame, holistic)
                        
                        # Utilizamos tu función maestra del config.py (coordenadas relativas a la nariz)
                        keypoints = extract_keypoints(results)
                        sequence_data.append(keypoints)

                    cap.release()

                    # Guardamos la secuencia matemática extraída en el .h5
                    if sequence_data:
                        hf.create_dataset(dataset_name, data=np.array(sequence_data))
                        print(f"Procesado: {dataset_name} | Longitud: {len(sequence_data)} frames")
                    else:
                        print(f"ADVERTENCIA: No se detectó información en {video_file}")

if __name__ == "__main__":
    print("Iniciando procesamiento por lotes (Batch Processing)...")
    process_videos_to_h5()
    print("\n[ÉXITO] Todos los videos han sido transformados a archivos .h5 listos para entrenar.")
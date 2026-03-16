import cv2
import numpy as np
import h5py
import mediapipe as mp
import os
from config import *

def draw_custom_keypoints(image, results):
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    
    # 1. Dibujar SOLO los 16 puntos faciales clave manualmente
    if results.face_landmarks:
        h, w, _ = image.shape 
        for idx in SELECTED_FACE_INDICES:
            landmark = results.face_landmarks.landmark[idx]
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (cx, cy), 3, (0, 255, 255), -1) 
            
    # 2. Dibujar Pose y Manos
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

def capture_dynamic_samples(word, target_samples=60):

    # --- NUEVA REGLA: Aumento automático para la clase "nada" ---
    if word.lower() == "nada":
        target_samples = int(target_samples * 1.75)
        print(f"\n[INFO] Clase 'nada' detectada. La meta se ajustó automáticamente a {target_samples} muestras.")

    create_folder_if_not_exists(DATA_PATH)
    file_path = os.path.join(DATA_PATH, f"{word}.h5")
    
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(0)
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Usamos 'a' (append) para leer lo que hay y poder escribir al mismo tiempo
        with h5py.File(file_path, 'a') as hf:
            
            existing_samples = len(hf.keys())
            
            # VALIDACIÓN: Si ya tenemos la cantidad deseada (o más), saltamos la palabra
            if existing_samples >= target_samples:
                print(f"[*] La palabra '{word.upper()}' ya tiene {existing_samples} muestras. Saltando...")
                cap.release()
                cv2.destroyAllWindows()
                return
            
            # Calcular cuántas faltan
            samples_to_record = target_samples - existing_samples
            print(f"\n--- RECOLECTANDO MUESTRAS PARA: {word.upper()} ---")
            print(f"Muestras existentes: {existing_samples} | Faltan: {samples_to_record} para llegar a la meta de {target_samples}.")
            
            for i in range(samples_to_record):
                # Calcular el índice real para no sobreescribir (ej. si hay 30, empezamos en sample_30)
                current_sample_idx = existing_samples + i
                sequence_data = []
                recording = False
                
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    
                    image, results = mediapipe_detection(frame, holistic)
                    draw_custom_keypoints(image, results)
                    
                    # UI: Mostramos el progreso real respecto a la meta total
                    cv2.putText(image, f"Palabra: {word} | Muestra: {current_sample_idx + 1}/{target_samples}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
                    if recording:
                        cv2.putText(image, "GRABANDO... (Presiona 's' para detener)", (10, 70), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        keypoints = extract_keypoints(results)
                        sequence_data.append(keypoints)
                    else:
                        cv2.putText(image, "Presiona 'r' para iniciar captura", (10, 70), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    cv2.imshow('Captura de Datos', image)
                    key = cv2.waitKey(10) & 0xFF
                    
                    if key == ord('r') and not recording:
                        recording = True
                    elif key == ord('s') and recording:
                        recording = False
                        
                        # Guardar usando el índice calculado
                        dataset_name = f"sample_{current_sample_idx}"
                        hf.create_dataset(dataset_name, data=np.array(sequence_data))
                        print(f"Muestra {current_sample_idx + 1} guardada con {len(sequence_data)} frames.")
                        break
                    elif key == ord('q'):
                        print("\nGrabación interrumpida por el usuario.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # La meta global. Si ya hay algunos, solamente completara hasta el maximo, no elimina, solo agrega
    # ACTUAL: 80
    META_MUESTRAS = 80 
    
    for word in WORDS:
        capture_dynamic_samples(word, target_samples=META_MUESTRAS)
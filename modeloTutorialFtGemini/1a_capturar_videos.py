import cv2
import os
import mediapipe as mp
from config import *


def draw_custom_keypoints(image, results):
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
    if not os.path.exists(VIDEOS_FOLDER):
        os.makedirs(VIDEOS_FOLDER)
    for word in WORDS:
        word_path = os.path.join(VIDEOS_FOLDER, word)
        if not os.path.exists(word_path):
            os.makedirs(word_path)

def capture_videos(word, target_samples=80):
    if word.lower() == "nada":
        target_samples = int(target_samples * 1.75)
        print(f"\n[INFO] Clase 'nada' detectada. Meta ajustada a {target_samples} videos.")

    word_folder = os.path.join(VIDEOS_FOLDER, word)
    
    # Contamos cuántos videos .avi ya existen en la carpeta de la palabra
    existing_videos = len([f for f in os.listdir(word_folder) if f.endswith('.avi')])

    if existing_videos >= target_samples:
        print(f"[*] La palabra '{word.upper()}' ya tiene {existing_videos} videos. Saltando...")
        return

    samples_to_record = target_samples - existing_videos
    print(f"\n--- RECOLECTANDO VIDEOS PARA: {word.upper()} ---")
    print(f"Videos existentes: {existing_videos} | Faltan: {samples_to_record}")

    cap = cv2.VideoCapture(0)
    # Obtener resolución de la cámara para guardar el video correctamente
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30.0 # Frames por segundo estándar
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # Codec compatible y eficiente

    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for i in range(samples_to_record):
            current_sample_idx = existing_videos + i
            recording = False
            out = None
            
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                # Procesamos el frame para mostrar los dibujos en la pantalla
                image, results = mediapipe_detection(frame, holistic)
                draw_custom_keypoints(image, results)
                
                cv2.putText(image, f"Palabra: {word} | Video: {current_sample_idx + 1}/{target_samples}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                if recording:
                    cv2.putText(image, "GRABANDO... (Presiona 's' para detener)", (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # IMPORTANTE: Guardamos el 'frame' crudo original, NO la 'image' que tiene los dibujos
                    out.write(frame)
                else:
                    cv2.putText(image, "Presiona 'r' para iniciar captura", (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('Captura de Videos de LESSA', image)
                key = cv2.waitKey(10) & 0xFF
                
                if key == ord('r') and not recording:
                    recording = True
                    # Inicializamos el grabador de video al presionar 'r'
                    video_path = os.path.join(word_folder, f"sample_{current_sample_idx}.avi")
                    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                    
                elif key == ord('s') and recording:
                    recording = False
                    out.release() # Cerramos y guardamos el archivo de video
                    print(f"Video {current_sample_idx + 1} guardado correctamente.")
                    break
                    
                elif key == ord('q'):
                    print("\nGrabación interrumpida por el usuario.")
                    if out is not None: out.release()
                    cap.release()
                    cv2.destroyAllWindows()
                    return

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    META_MUESTRAS = 10 
    create_video_folders()
    for word in WORDS:
        capture_videos(word, target_samples=META_MUESTRAS)
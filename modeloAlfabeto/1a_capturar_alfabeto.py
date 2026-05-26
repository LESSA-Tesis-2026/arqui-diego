import cv2
import os
import time
from config_alfabeto import *

def draw_landmarks(image, results):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )

    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )

    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
        )

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
        )

def capture_images(letter, target_images=200, delay_seconds=0.1):
    letter_folder = os.path.join(DATASET_FOLDER, letter)
    create_folder_if_not_exists(letter_folder)
    
    existing_images = len([f for f in os.listdir(letter_folder) if f.endswith('.jpg')])
    if existing_images >= target_images:
        print(f"[*] La letra '{letter}' ya tiene {existing_images} imágenes. Saltando...")
        return

    images_to_capture = target_images - existing_images
    print(f"\n--- RECOLECTANDO IMÁGENES PARA: {letter} ---")
    
    cap = cv2.VideoCapture(0)
    recording = False
    last_capture_time = time.time()
    current_idx = existing_images

    with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while current_idx < target_images:
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)

            display_frame = frame.copy()
            draw_landmarks(display_frame, results)

            cv2.putText(
                display_frame,
                f"Letra: {letter} | Capturas: {current_idx}/{target_images}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

            if recording:
                cv2.putText(
                    display_frame,
                    "CAPTURANDO... (Mueve la mano ligeramente)",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

                remaining = max(0.0, delay_seconds - (time.time() - last_capture_time))
                cv2.putText(
                    display_frame,
                    f"Siguiente captura en: {remaining:.1f}s",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

                # Lógica del temporizador
                if time.time() - last_capture_time > delay_seconds:
                    img_path = os.path.join(letter_folder, f"img_{current_idx}.jpg")
                    cv2.imwrite(img_path, frame)  # Guardamos el frame LIMPIO
                    current_idx += 1
                    last_capture_time = time.time()

                    # Efecto visual de flash
                    cv2.rectangle(
                        display_frame,
                        (0, 0),
                        (frame.shape[1], frame.shape[0]),
                        (255, 255, 255),
                        -1,
                    )
            else:
                cv2.putText(
                    display_frame,
                    "Presiona 'r' para iniciar auto-captura",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow('Captura de Alfabeto LESSA', display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):
                recording = True
            elif key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    META_IMAGENES = 60 # Recomendado para estáticos
    TIEMPO_ESPERA = 1.0 # Toma una foto cada 0.1 segundos
    
    create_folder_if_not_exists(DATASET_FOLDER)
    for letter in ALPHABET:
        capture_images(letter, target_images=META_IMAGENES, delay_seconds=TIEMPO_ESPERA)
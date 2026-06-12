import cv2
import numpy as np
import mediapipe as mp
import collections
from tensorflow.keras.models import load_model
from config_alfabeto import *

def draw_keypoints(image, results):
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def real_time_alphabet(threshold=0.80):
    model = load_model(MODEL_PATH)
    mp_holistic = mp.solutions.holistic
    
    # Sistema de Suavizado (Debouncing)
    BUFFER_SIZE = 5
    predictions_buffer = collections.deque(maxlen=BUFFER_SIZE)
    current_letter = "-"
    
    cap = cv2.VideoCapture(0)
    
    with mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            image, results = mediapipe_detection(frame, holistic)
            draw_keypoints(image, results)
            
            if results.left_hand_landmarks or results.right_hand_landmarks:
                # 1. Extraer matriz del frame actual (1D array de 306)
                keypoints = extract_keypoints(results)
                
                # 2. Reshape para Keras (1, 306)
                keypoints_reshaped = np.expand_dims(keypoints, axis=0)
                
                # 3. Predicción
                res = model.predict(keypoints_reshaped, verbose=0)[0]
                best_match_idx = np.argmax(res)
                
                if res[best_match_idx] > threshold:
                    predicted_char = ALPHABET[best_match_idx]
                else:
                    predicted_char = "-"
                    
                predictions_buffer.append(predicted_char)
                
                # 4. Votación para estabilizar la pantalla
                word_counts = collections.Counter(predictions_buffer)
                most_common, count = word_counts.most_common(1)[0]
                if count >= 3: # Si 3 de los últimos 5 frames coinciden
                    current_letter = most_common
            else:
                current_letter = "-"
                predictions_buffer.clear()

            # --- INTERFAZ ---
            cv2.rectangle(image, (0, 0), (640, 60), (245, 117, 16), -1)
            cv2.putText(image, f"LETRA: {current_letter}", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
            
            cv2.imshow('Traductor Alfabeto LESSA', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_alphabet(threshold=0.80)
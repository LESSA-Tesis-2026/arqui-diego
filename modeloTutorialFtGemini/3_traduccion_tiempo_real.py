import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import *

# CSLR - Continuous Sign Language Recognition
# TODO: buscar directamente esto

def draw_keypoints(image, results):
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def real_time_translation(threshold=0.70):
    model = load_model(MODEL_PATH)
    mp_holistic = mp.solutions.holistic
    
    sequence = []
    sentence = []
    current_probs = np.zeros(len(WORDS))
    
    # --- PARÁMETROS DE TRADUCCIÓN CONTINUA ---
    # 1. Ventana Deslizante: Cuántos frames recientes forman una seña. 
    # (30 frames = aprox 1 segundo de movimiento fluido)
    REAL_TIME_WINDOW = 20 
    
    # 2. Votación Continua (Estabilidad)
    prediction_buffer = []
    VOTING_WINDOW = 7       # Memoria de las últimas 15 predicciones
    VOTES_REQUIRED = 3      # Exigimos consenso para evitar errores
    
    # 3. Enfriamiento (Evitar repeticiones indeseadas)
    cooldown_counter = 0     
    COOLDOWN_FRAMES = 15     # Frames que espera antes de aceptar otra seña
    last_word = ""           # Rastrea la última palabra impresa
    
    cap = cv2.VideoCapture(0)
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            image, results = mediapipe_detection(frame, holistic)
            draw_keypoints(image, results)
            
            # 1. ACUMULACIÓN CONTINUA (El flujo nunca se limpia por completo)
            if results.left_hand_landmarks or results.right_hand_landmarks:
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
            else:
                # Si bajamos las manos, inyectamos ceros para "diluir" la ventana
                sequence.append(np.zeros(LENGTH_KEYPOINTS))
            
            # La ventana siempre avanza, manteniendo estrictamente los últimos X frames
            sequence = sequence[-REAL_TIME_WINDOW:]
            
            # 2. LÓGICA DE PREDICCIÓN CONTINUA
            if len(sequence) == REAL_TIME_WINDOW:
                
                # TRUCO DE PADDING: Tomamos los 30 frames reales y los llevamos a 60
                # Esto alinea los datos para que sean idénticos a los del entrenamiento
                pad_seq = pad_sequences([sequence], maxlen=MAX_FRAMES, padding='post', dtype='float32')
                res = model.predict(pad_seq, verbose=0)[0]
                current_probs = res 
                
                best_match_idx = np.argmax(res)
                
                # Asignar la palabra si supera la confianza
                if res[best_match_idx] > threshold:
                    current_word = WORDS[best_match_idx]
                else:
                    current_word = "ninguna"
                    
                prediction_buffer.append(current_word)
                prediction_buffer = prediction_buffer[-VOTING_WINDOW:]
                
                # 3. LÓGICA DE EMISIÓN Y ENFRIAMIENTO
                if cooldown_counter > 0:
                    cooldown_counter -= 1
                else:
                    # Si detectamos una palabra con suficientes votos
                    if current_word != "ninguna" and prediction_buffer.count(current_word) >= VOTES_REQUIRED:
                        
                        # Imprimir solo si es diferente a la última seña que hicimos
                        if current_word != last_word:
                            sentence.append(current_word)
                            last_word = current_word
                            cooldown_counter = COOLDOWN_FRAMES # Activar enfriamiento
                            
                            if len(sentence) > 5:
                                sentence = sentence[-5:]
                
                # Reseteamos la 'última palabra' si el usuario se queda en reposo
                # Esto permite que pueda repetir la misma palabra (ej. "gracias", "gracias") si hace una pausa
                if prediction_buffer.count("ninguna") >= 10:
                    last_word = ""
            
            # 4. INTERFAZ GRÁFICA (UI)
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
            
            cv2.imshow('Traductor LESSA', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_translation(threshold=0.65)
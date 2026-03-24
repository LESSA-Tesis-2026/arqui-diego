import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import collections
from config import *

def draw_keypoints(image, results):
    # Usamos la función de dibujo estándar de MediaPipe para la interfaz
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def real_time_translation(threshold=0.75):
    model = load_model(MODEL_PATH)
    mp_holistic = mp.solutions.holistic
    
    # 1. PARÁMETROS CSLR (Ajustables según la velocidad de tus señas)
    WINDOW_SIZE = 40           # Tamaño de la ventana deslizante (aprox. 1 segundo de video)
    VOTING_BUFFER_SIZE = 15    # Historial de predicciones para el suavizado
    MIN_VOTES = 10             # Votos necesarios para confirmar una palabra
    
    # 2. ESTRUCTURAS DE DATOS CONTINUAS
    sequence = collections.deque(maxlen=WINDOW_SIZE)
    predictions_buffer = collections.deque(maxlen=VOTING_BUFFER_SIZE)
    
    # 3. MÁQUINA DE ESTADOS
    sentence = []
    last_emitted_word = "nada"
    nada_counter = 0           # Cuenta cuánto tiempo llevamos en reposo
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
                # Utilizamos la función de config.py que ya calcula coordenadas relativas a la nariz
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
            else:
                # Si las manos salen de cámara, inyectamos ceros para mantener el flujo de tiempo real
                sequence.append(np.zeros(LENGTH_KEYPOINTS))
            
            # --- FASE 2: PREDICCIÓN CONTINUA ---
            if len(sequence) == WINDOW_SIZE:
                # El modelo espera MAX_FRAMES (60). Rellenamos nuestra ventana de 40 con ceros al final.
                # La capa Masking ignorará este relleno matemático.
                pad_seq = pad_sequences([list(sequence)], maxlen=MAX_FRAMES, padding='post', dtype='float32')
                
                res = model.predict(pad_seq, verbose=0)[0]
                current_probs = res 
                best_match_idx = np.argmax(res)
                
                if res[best_match_idx] > threshold:
                    current_word = WORDS[best_match_idx]
                else:
                    current_word = "nada"
                
                # Agregamos la predicción al búfer de votación
                predictions_buffer.append(current_word)
                
                # --- FASE 3: SUAVIZADO Y LÓGICA DE TRANSICIÓN ---
                # Validamos cuál es la palabra más repetida en el último instante de tiempo
                word_counts = collections.Counter(predictions_buffer)
                most_common_word, count = word_counts.most_common(1)[0]
                
                if count >= MIN_VOTES:
                    stable_word = most_common_word
                else:
                    stable_word = "nada"
                
                # Máquina de estados para emitir la palabra a la oración
                if stable_word != "nada":
                    nada_counter = 0 # Reiniciamos el contador de reposo
                    
                    # Solo agregamos si es una palabra nueva (evita "hola hola hola")
                    if stable_word != last_emitted_word:
                        sentence.append(stable_word)
                        last_emitted_word = stable_word
                        
                        # Mantenemos la oración en un máximo de 5 palabras para no saturar la pantalla
                        if len(sentence) > 5:
                            sentence = sentence[-5:]
                            
                else:
                    # Si detectamos "nada" constantemente, permitimos que el usuario repita la última palabra
                    nada_counter += 1
                    if nada_counter > 15: # Medio segundo de pausa real
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
            
            cv2.imshow('Traductor CSLR LESSA', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_translation(threshold=0.75)
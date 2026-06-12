import cv2
import numpy as np
import mediapipe as mp
import collections
import textwrap
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Importamos ambas configuraciones usando alias para evitar colisión de nombres
import modeloTutorialFtGemini.config as cfg_words
import modeloAlfabeto.config_alfabeto as cfg_alfabeto

def draw_keypoints(image, results):
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def real_time_combined(threshold_words=0.65, threshold_alphabet=0.80):
    print("Cargando modelo de Palabras...")
    model_words = load_model(cfg_words.MODEL_PATH)
    print("Cargando modelo de Alfabeto...")
    model_alphabet = load_model(cfg_alfabeto.MODEL_PATH)
    
    mp_holistic = mp.solutions.holistic
    
    # --- HIPERPARÁMETROS Y VARIABLES DE ESTADO ---
    current_mode = "words" 
    
    # TEXTO GLOBAL (Donde se guarda todo)
    accumulated_text = ""
    
    # Estado para Palabras
    WINDOW_SIZE = 25           
    VOTING_BUFFER_SIZE_WORDS = 10    
    MIN_VOTES_WORDS = 7             
    sequence = collections.deque(maxlen=WINDOW_SIZE)
    predictions_buffer_words = collections.deque(maxlen=VOTING_BUFFER_SIZE_WORDS)
    last_emitted_word = "nada"
    nada_counter = 0           
    current_probs = np.zeros(len(cfg_words.WORDS))
    
    # Estado para Alfabeto
    BUFFER_SIZE_ALFABETO = 5
    predictions_buffer_alphabet = collections.deque(maxlen=BUFFER_SIZE_ALFABETO)
    last_emitted_letter = "-"  # Para evitar que se repita infinitamente la misma letra
    
    cap = cv2.VideoCapture(0)
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            image, results = cfg_words.mediapipe_detection(frame, holistic)
            draw_keypoints(image, results)
            
            # ==========================================
            # MODO: TRADUCCIÓN DE PALABRAS (CSLR)
            # ==========================================
            if current_mode == "words":
                if results.left_hand_landmarks or results.right_hand_landmarks:
                    keypoints = cfg_words.extract_keypoints(results)
                    sequence.append(keypoints)
                else:
                    sequence.append(np.zeros(cfg_words.LENGTH_KEYPOINTS))
                
                if len(sequence) == WINDOW_SIZE:
                    seq_array = np.array(sequence)
                    delta = np.vstack([seq_array[0:1, :], np.diff(seq_array, axis=0)])
                    delta_delta = np.vstack([delta[0:1, :], np.diff(delta, axis=0)])
                    combined_seq = np.concatenate([seq_array, delta, delta_delta], axis=-1)
                    
                    pad_seq = pad_sequences([combined_seq], maxlen=cfg_words.MAX_FRAMES, padding='post', dtype='float32')
                    res = model_words.predict(pad_seq, verbose=0)[0]
                    current_probs = res 
                    best_match_idx = np.argmax(res)
                    
                    if res[best_match_idx] > threshold_words:
                        current_word = cfg_words.WORDS[best_match_idx]
                    else:
                        current_word = "nada"
                    
                    predictions_buffer_words.append(current_word)
                    
                    word_counts = collections.Counter(predictions_buffer_words)
                    most_common_word, count = word_counts.most_common(1)[0]
                    
                    if count >= MIN_VOTES_WORDS: stable_word = most_common_word
                    else: stable_word = "nada"
                    
                    if stable_word != "nada":
                        nada_counter = 0 
                        if stable_word != last_emitted_word:
                            # Añadir espacio si el texto no está vacío y no termina ya en espacio
                            if len(accumulated_text) > 0 and not accumulated_text.endswith(" "):
                                accumulated_text += " "
                            accumulated_text += stable_word + " "
                            last_emitted_word = stable_word
                    else:
                        nada_counter += 1
                        if nada_counter > 15: 
                            last_emitted_word = "nada"

                # Interfaz de probabilidades para palabras (movida más abajo para dar espacio al texto)
                panel_height = 110 + (len(cfg_words.WORDS) * 30)
                cv2.rectangle(image, (0, 110), (250, panel_height), (40, 40, 40), -1)
                
                y_offset = 140
                for i, word in enumerate(cfg_words.WORDS):
                    prob = current_probs[i]
                    color = (0, 255, 0) if (prob == np.max(current_probs) and prob > threshold_words) else (255, 255, 255)
                    text = f"{word}: {prob*100:.1f}%"
                    cv2.putText(image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
                    y_offset += 30

            # ==========================================
            # MODO: TRADUCCIÓN DE ALFABETO
            # ==========================================
            elif current_mode == "alphabet":
                if results.left_hand_landmarks or results.right_hand_landmarks:
                    keypoints = cfg_alfabeto.extract_keypoints(results)
                    keypoints_reshaped = np.expand_dims(keypoints, axis=0)
                    
                    res = model_alphabet.predict(keypoints_reshaped, verbose=0)[0]
                    best_match_idx = np.argmax(res)
                    
                    if res[best_match_idx] > threshold_alphabet:
                        predicted_char = cfg_alfabeto.ALPHABET[best_match_idx]
                    else:
                        predicted_char = "-"
                        
                    predictions_buffer_alphabet.append(predicted_char)
                    
                    word_counts = collections.Counter(predictions_buffer_alphabet)
                    most_common, count = word_counts.most_common(1)[0]
                    
                    if count >= 3:
                        current_letter = most_common
                    else:
                        current_letter = "-"
                        
                    # Lógica para pegar letras juntas (H + O + L + A)
                    if current_letter != "-" and current_letter != last_emitted_letter:
                        accumulated_text += current_letter
                        last_emitted_letter = current_letter
                    elif current_letter == "-":
                        # Si dejamos de hacer la seña, reseteamos la última letra
                        # Esto permite escribir letras repetidas (ej. 'L' luego soltar y luego 'L' para "LL")
                        last_emitted_letter = "-"
                        
                else:
                    last_emitted_letter = "-"
                    predictions_buffer_alphabet.clear()

            # ==========================================
            # INTERFAZ GRÁFICA COMPARTIDA (TEXTO PERSISTENTE)
            # ==========================================
            
            # Fondo del texto (más grande para permitir múltiples líneas)
            cv2.rectangle(image, (0, 0), (640, 110), (245, 117, 16), -1)
            
            # Envolver el texto para que no se salga de la pantalla (aprox 35 caracteres por línea)
            wrapped_lines = textwrap.wrap(accumulated_text, width=35)
            
            # Imprimir solo las últimas 3 líneas para que no se desborde el recuadro
            y_text = 35
            for line in wrapped_lines[-3:]:
                cv2.putText(image, line.upper(), (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                y_text += 35

            # --- INDICADOR GENERAL DE MODO ---
            mode_text = "MODO: PALABRAS" if current_mode == "words" else "MODO: ALFABETO"
            cv2.putText(image, f"{mode_text} ('s' = cambiar | 'b' = borrar)", (10, image.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('Traductor Híbrido LESSA', image)
            
            # --- CONTROL DE TECLADO ---
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Alternar modo
                if current_mode == "words":
                    current_mode = "alphabet"
                    predictions_buffer_alphabet.clear()
                    last_emitted_letter = "-"
                    # Opcional: Agregar un espacio al pasar a letras si no hay uno
                    if len(accumulated_text) > 0 and not accumulated_text.endswith(" "):
                        accumulated_text += " "
                else:
                    current_mode = "words"
                    sequence.clear()
                    predictions_buffer_words.clear()
                    last_emitted_word = "nada"
            elif key == ord('b'):
                # Limpiar todo el texto
                accumulated_text = ""
                last_emitted_word = "nada"
                last_emitted_letter = "-"
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_combined()
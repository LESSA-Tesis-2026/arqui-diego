import cv2
import numpy as np
import mediapipe as mp
import collections
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Importamos las variables de ambos archivos de configuración
# Usamos alias para no confundir las rutas de los modelos
from modeloTutorialFtGemini.config import WORDS, MAX_FRAMES, MODEL_PATH as RUTA_MODELO_PALABRAS, LENGTH_KEYPOINTS as LENGTH_KEYPOINTS_WORDS, extract_keypoints as extract_keypoints_words, mediapipe_detection
from modeloAlfabeto.config_alfabeto import ALPHABET, MODEL_PATH as RUTA_MODELO_LETRAS, extract_keypoints as extract_keypoints_letters

def wrap_text_to_width(text, max_width, font, font_scale, thickness):
    if not text:
        return [""]

    tokens = text.split(" ")
    lines = []
    current = ""

    for token in tokens:
        candidate = token if current == "" else f"{current} {token}"
        if cv2.getTextSize(candidate, font, font_scale, thickness)[0][0] <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)
            current = ""

        for ch in token:
            candidate = ch if current == "" else f"{current}{ch}"
            if cv2.getTextSize(candidate, font, font_scale, thickness)[0][0] <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = ch

    if current:
        lines.append(current)

    return lines

def draw_keypoints(image, results):
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def unified_real_time_translation(word_threshold=0.80, letter_threshold=0.85, motion_threshold=0.005):
    print("Cargando cerebro dinámico (Palabras LSTM)...")
    modelo_palabras = load_model(RUTA_MODELO_PALABRAS)
    
    print("Cargando cerebro estático (Alfabeto Denso)...")
    modelo_letras = load_model(RUTA_MODELO_LETRAS)
    print("Modelos cargados exitosamente. Iniciando cámara...")

    mp_holistic = mp.solutions.holistic
    
    WINDOW_SIZE = 25           
    sequence = collections.deque(maxlen=WINDOW_SIZE)
    
    word_buffer = collections.deque(maxlen=10)
    letter_buffer = collections.deque(maxlen=5)
    
    sentence = []
    last_emitted_item = ""
    nada_counter = 0 
    
    # Contadores de estabilidad (El "Embrague")
    dynamic_frames = 0
    static_frames = 0          
    
    estado_actual = "BUSCANDO MANOS"
    current_word = "-"
    current_letter = "-"
    
    cap = cv2.VideoCapture(0)
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            image, results = mediapipe_detection(frame, holistic)
            #draw_keypoints(image, results)
            
            manos_detectadas = bool(results.left_hand_landmarks or results.right_hand_landmarks)
            
            if manos_detectadas:
                keypoints_words = extract_keypoints_words(results)
                keypoints_letters = extract_keypoints_letters(results)
                sequence.append(keypoints_words)
                nada_counter = 0
            else:
                keypoints_letters = None
                sequence.append(np.zeros(LENGTH_KEYPOINTS_WORDS))
                estado_actual = "SIN MANOS"
                word_buffer.clear()
                letter_buffer.clear()
                nada_counter += 1
            
            # --- SEMÁFORO DE VELOCIDAD MEJORADO ---
            if len(sequence) == WINDOW_SIZE and manos_detectadas:
                seq_array = np.array(sequence)
                delta = np.vstack([seq_array[0:1, :], np.diff(seq_array, axis=0)])
                
                # Promediamos 5 frames para ignorar el temblor de la mano
                velocidad_manos = np.mean(np.abs(delta[-5:, 180:]))
                
                # --- MODO PALABRAS (Movimiento) ---
                if velocidad_manos > motion_threshold:
                    dynamic_frames += 1
                    static_frames = 0
                    estado_actual = f"PALABRA (Vel: {velocidad_manos:.4f})"
                    
                    # Si el movimiento es sostenido (transición), reseteamos las letras
                    if dynamic_frames > 4:
                        letter_buffer.clear()
                        # Permite escribir letras dobles (ej: "E" -> "E") al detectar que moviste la mano
                        if len(last_emitted_item) == 1: 
                            last_emitted_item = ""
                    
                    # Ejecutamos LSTM (306 + delta + delta-delta)
                    delta_delta = np.vstack([delta[0:1, :], np.diff(delta, axis=0)])
                    combined_seq = np.concatenate([seq_array, delta, delta_delta], axis=-1)
                    pad_seq = pad_sequences([combined_seq], maxlen=MAX_FRAMES, padding='post', dtype='float32')
                    
                    res_word = modelo_palabras.predict(pad_seq, verbose=0)[0]
                    best_idx = np.argmax(res_word)
                    
                    if res_word[best_idx] > word_threshold:
                        predicted_word = WORDS[best_idx]
                        current_word = f"{predicted_word} ({res_word[best_idx]*100:.0f}%)"
                    else:
                        predicted_word = "nada"
                        current_word = "-"

                    word_buffer.append(predicted_word)
                        
                # --- MODO LETRAS (Estático) ---
                else:
                    static_frames += 1
                    dynamic_frames = 0
                    estado_actual = f"LETRA (Vel: {velocidad_manos:.4f})"
                    
                    # Si nos quedamos quietos, borramos rastro de palabras a medias
                    if static_frames > 4:
                        word_buffer.clear()
                    
                    # Ejecutamos Densa
                    if keypoints_letters is not None:
                        ultimo_frame = np.expand_dims(keypoints_letters, axis=0)
                        res_letter = modelo_letras.predict(ultimo_frame, verbose=0)[0]
                        best_idx = np.argmax(res_letter)

                        if res_letter[best_idx] > letter_threshold:
                            letter_buffer.append(ALPHABET[best_idx])
                            current_letter = f"{ALPHABET[best_idx]} ({res_letter[best_idx]*100:.0f}%)"
                        else:
                            letter_buffer.append("-")
                            current_letter = "-"
                    else:
                        letter_buffer.clear()
                        current_letter = "-"

            # --- MÁQUINA DE ESTADOS ESTRICTA ---
            # 1. Impresión de Palabras (segun script original)
            if len(word_buffer) > 0:
                counts = collections.Counter(word_buffer)
                most_common, votes = counts.most_common(1)[0]
                if votes >= 7 and most_common != "nada" and most_common != last_emitted_item:
                    sentence.append(f" {most_common} ")
                    last_emitted_item = most_common

            # 2. Impresión de Letras (Rápida y responsiva)
            if len(letter_buffer) > 0:
                counts = collections.Counter(letter_buffer)
                most_common, votes = counts.most_common(1)[0]
                if votes >= 3 and most_common != "-" and most_common != last_emitted_item:
                    sentence.append(most_common)
                    last_emitted_item = most_common

            # Limpiar si bajamos las manos mucho tiempo
            if nada_counter > 20:
                last_emitted_item = ""

            # --- INTERFAZ GRÁFICA ---
            box_x0, box_y0 = 0, 0
            box_x1, box_y1 = 640, 120
            cv2.rectangle(image, (box_x0, box_y0), (box_x1, box_y1), (245, 117, 16), -1)

            display_text = "".join(sentence)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.9
            thickness = 2
            line_height = cv2.getTextSize("Ag", font, font_scale, thickness)[0][1] + 8
            max_width = box_x1 - box_x0 - 20
            max_lines = max(1, (box_y1 - box_y0 - 10) // line_height)

            lines = wrap_text_to_width(display_text.upper(), max_width, font, font_scale, thickness)
            lines = lines[-max_lines:]

            y = box_y0 + 30
            for line in lines:
                cv2.putText(image, line, (10, y), font, font_scale, (255, 255, 255), thickness)
                y += line_height
            
            cv2.rectangle(image, (0, 130), (320, 250), (40, 40, 40), -1)
            color_estado = (0, 255, 0) if "LETRA" in estado_actual else (0, 165, 255)
            if "BUSCANDO" in estado_actual or "SIN" in estado_actual: color_estado = (150, 150, 150)
            
            cv2.putText(image, f"MODO: {estado_actual}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_estado, 1)
            cv2.putText(image, f"Pred. Palabra: {current_word}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(image, f"Pred. Letra:   {current_letter}", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Traductor Hibrido LESSA', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Umbrales reajustados para máxima estabilidad
    unified_real_time_translation(word_threshold=0.70, letter_threshold=0.80, motion_threshold=0.001)
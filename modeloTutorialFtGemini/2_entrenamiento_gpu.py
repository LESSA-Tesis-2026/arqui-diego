import os
import h5py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, GRU, SpatialDropout1D, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l2
from config import *

# Habilitar memoria dinámica para la GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def data_augmentation(X_train, y_train):
    aug_sequences, aug_labels = [], []
    for seq, label in zip(X_train, y_train):
        # 1. Original
        aug_sequences.append(seq)
        aug_labels.append(label)
        
        # 2. Ruido MUY suave (simula pequeños errores de cámara)
        noise_1 = np.random.normal(0, 0.005, seq.shape)
        aug_sequences.append(seq + noise_1)
        aug_labels.append(label)

        # 3. Ruido moderado (pero menor que antes)
        noise_2 = np.random.normal(0, 0.010, seq.shape)
        aug_sequences.append(seq + noise_2)
        aug_labels.append(label)
        
    return aug_sequences, aug_labels

def load_raw_data_from_h5():
    sequences, labels = [], []
    for label, word in enumerate(WORDS):
        file_path = os.path.join(DATA_PATH, f"{word}.h5")
        if not os.path.exists(file_path): continue
        
        with h5py.File(file_path, 'r') as hf:
            for key in hf.keys():
                seq = np.array(hf[key])
                sequences.append(seq)
                labels.append(label)
                
    return sequences, labels

# CAMBIOS

def build_model():
    model = Sequential([
 
        # Masking ignora los frames con 0s, los cuales se han utilizado para rellenar y estandarizar todas las secuencias con la misma longitud
        
        Masking(mask_value=0.0, input_shape=(MAX_FRAMES, LENGTH_KEYPOINTS)),
        
        # SpatialDropout apaga canales enteros (ej. "ciega" a la red de la coordenada Z por un rato; ignora la profundidad)
        SpatialDropout1D(0.3), 
        
        # Cambiamos a LSTM Bidireccional y aplicamos castigo L2

        # - LSTM bidireccional: Procesa la secuencia tanto hacia adelante como hacia atrás, lo que puede ayudar
        # a capturar mejor las dependencias temporales en ambas direcciones. Especialmente en casos donde el inicio o el fin de 2 señas
        # diferentes pueden ser similares, el contexto completo de la secuencia ayuda a diferenciarlas.
        # - L2: Penaliza los pesos grandes, lo que ayuda a reducir el overfitting
        # - Dropout: Apaga neuronas aleatorias durante el entrenamiento para evitar que la red dependa demasiado de ciertas características,
        # lo que también ayuda a combatir el overfitting ya que evita que memorice ciertas caracteristicas.

        Bidirectional(LSTM(64, return_sequences=True, activation='tanh', kernel_regularizer=l2(0.0005))),
        Dropout(0.3), # Aumentamos el Dropout para combatir el overfitting

        Bidirectional(LSTM(32, return_sequences=False, activation='tanh', kernel_regularizer=l2(0.0005))),
        Dropout(0.3), # Aumentamos el Dropout para combatir el overfitting
        
        # Capa de condensación

        # - Dense Relu: Capa densa con activación ReLU para introducir no linealidad. La función ReLU es eficiente y ayuda a
        # la red a aprender patrones complejos; Toma toda la información temporal que extrajeron las LSTM y la aplana en
        # características lógicas simples.
        # - Dense softmax: Capa de salida con activación softmax para clasificación multiclase. Convierte las salidas anteriores en probabilidades

        Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(len(WORDS), activation='softmax')
    ])
    
    # Reducimos ligeramente el weight_decay para compensar el L2 nuevo
    optimizer = AdamW(learning_rate=0.0005, weight_decay=0.0005)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model

def plot_metrics(history, y_true, y_pred_classes):
    create_folder_if_not_exists(METRICS_FOLDER)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Precisión')
    ax1.legend()
    
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Pérdida')
    ax2.legend()
    plt.savefig(os.path.join(METRICS_FOLDER, 'training_history.png'))
    plt.close()

    cm = confusion_matrix(y_true, y_pred_classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=WORDS, yticklabels=WORDS)
    plt.title('Matriz de Confusión Normalizada')
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción')
    plt.savefig(os.path.join(METRICS_FOLDER, 'confusion_matrix.png'))
    plt.close()

if __name__ == "__main__":
    X_raw, y_raw = load_raw_data_from_h5()
    print(f"Total de muestras reales capturadas: {len(X_raw)}")
    
    # División 80/20. No haremos set de Test hasta que tengas al menos 100 muestras por palabra.
    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )
    
    X_train_aug, y_train_aug = data_augmentation(X_train_raw, y_train_raw)
    print(f"Muestras de entrenamiento tras aumentación: {len(X_train_aug)}")
    print(f"Muestras de validación (intactas): {len(X_val_raw)}")
    
    X_train = pad_sequences(X_train_aug, maxlen=MAX_FRAMES, padding='post', truncating='post', dtype='float32')
    X_val = pad_sequences(X_val_raw, maxlen=MAX_FRAMES, padding='post', truncating='post', dtype='float32')
    
    y_train = tf.keras.utils.to_categorical(y_train_aug, num_classes=len(WORDS))
    y_val = tf.keras.utils.to_categorical(y_val_raw, num_classes=len(WORDS))

    model = build_model()
    model.summary()

    create_folder_if_not_exists(MODEL_FOLDER_PATH)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)

    # - batch size: Aumentar el batch size puede ayudar a estabilizar el entrenamiento y aprovechar mejor la GPU, pero también puede requerir más memoria.
    # Dicta cuántos videos ve la red antes de actualizar su conocimiento, calcula el error promedio de sus predicciones en ese grupo, y da un solo paso
    # matemático para corregir sus pesos.
    # - epochs: Aumentar el número de epochs permite que el modelo tenga más oportunidades para aprender, pero también aumenta el riesgo de overfitting.
    # - EarlyStopping: El entrenamiento se detendrá automáticamente si el modelo deja de mejorar en el conjunto de validación.

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=16,
        callbacks=[early_stop, checkpoint, reduce_lr]
    )

    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_val, axis=1)

    print("\n--- REPORTE DE CLASIFICACIÓN ---")
    print(classification_report(y_true, y_pred_classes, target_names=WORDS))

    plot_metrics(history, y_true, y_pred_classes)

# si se extiende el modelo, hay q utilizar normalizacion con "LayerNormalization"
#Cuando se escale a un modelo mas grande tomar en consideracion el agregar las siguientes capas:
#
#- Capas Convolucionales 1D 
#a. ¿Qué hacen? 
#Las capas convolucionales son famosas en imágenes (2D), pero en 1D actúan como un "escáner de tiempo". 
#En lugar de mirar todo el video a la vez, una Conv1D agrupa pequeños bloques de tiempo (por ejemplo, de 3 en 3 frames) y
#extrae "micro-patrones" (una aceleración repentina de la mano, un cierre rápido de dedos).
#
#b. ¿Por qué agregarla? 
#Las LSTM son increíbles para entender el inicio y el fin de una oración, pero son malas procesando detalles rápidos. 
#Al poner una o dos capas Conv1D antes de tus LSTM, las convoluciones procesan los 306 puntos crudos, extraen las características
#cinemáticas más finas y le entregan a la LSTM un "resumen masticado" mucho más rico.
#
#c. El impacto: 
#Reduce drásticamente el esfuerzo de la LSTM, baja los tiempos de entrenamiento y captura esos micromovimientos que diferencian
#dos señas casi idénticas.
#--------------------------------------------------------------------------------------------------------------------------------------------------
#- NormalizationLayer
#a. ¿Qué hacen? 
#Normalizan las activaciones matemáticas de las neuronas, pero lo hacen frame por frame de manera independiente.
#
#b. ¿Por qué agregarla?
#Al escalar a 100 palabras, tu red neuronal tendrá que ser mucho más profunda y ancha (quizás pases de 600,000 parámetros a 2 o 3 millones).
#En redes tan grandes, los valores matemáticos tienden a dispararse o a encogerse hasta desaparecer (desvanecimiento del gradiente). 
#LayerNormalization actúa como un regulador de voltaje que mantiene las matemáticas estables, permitiendo que la red profunda aprenda sin colapsar.
#
#c. El impacto:
#Permite entrenar modelos mucho más grandes sin que se vuelvan inestables, lo que es crucial para manejar un vocabulario de 100 palabras.
#--------------------------------------------------------------------------------------------------------------------------------------------------
#- Mecanismo de Atención (Attention o Self-Attention)
#a. ¿Qué hace? 
#Le enseña a la red a "ignorar" el tiempo muerto y concentrarse en el clímax del movimiento. Le asigna un peso matemático (de 0 a 1) a cada
#frame del video.
#
#b. ¿Por qué agregarlo? 
#Imagina un video de 60 frames donde la seña ocurre entre el frame 20 y el 40. Las LSTM tratan todos los frames por igual, 
#lo que diluye la información. Una capa de Atención aprende que el frame 35 tiene la clave de la palabra y lo multiplica dándole la
#máxima prioridad, ignorando el reposo del inicio y el final.
#
#c. El impacto: 
#Es la diferencia entre un modelo bueno y uno de estado del arte. Permite que la red se concentre en la forma exacta de la mano en el punto
#máximo de la seña.
#---------------------------------------------------------------------------------------------------------------------------------------------------
#- Aumento de Datos Avanzado
#a. ¿Qué es?
#Además de agregar ruido, puedes hacer transformaciones más sofisticadas como invertir la secuencia (reproducir el video al revés),
#intercambiar manos (simular que la seña se hace con la mano izquierda en lugar de la derecha), o incluso usar técnicas de GANs para generar videos sintéticos.
#
#b. ¿Por qué agregarlo?
#Con 100 palabras, es difícil capturar suficientes variaciones de cada seña. Estas técnicas avanzadas pueden multiplicar tu dataset sin necesidad de
#grabar horas adicionales de video, ayudando a la red a generalizar mejor.
#
#c. El impacto:
#Puede marcar la diferencia entre un modelo que solo funciona con tus videos de entrenamiento y uno que generaliza bien a nuevos usuarios y 
#condiciones de grabación.
#---------------------------------------------------------------------------------------------------------------------------------------------------
#-Conexiones Residuales (Skip Connections o Add)
#a. ¿Qué hacen? 
#Crean "atajos" en la red neuronal. Toman los datos originales que entraron a una capa y se los suman directamente a la salida de esa capa.
#
#b. ¿Por qué agregarlo? 
#Cuando apilas muchas capas (ej. Conv1D -> LSTM -> LSTM -> Dense), la información original a veces se deforma tanto que la última capa ya no
#sabe qué estaba mirando. El atajo asegura que la red no olvide la posición original de las manos, mejorando la precisión en vocabularios gigantes.

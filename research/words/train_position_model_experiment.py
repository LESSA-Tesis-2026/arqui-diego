"""Entrena un experimento de palabras/frases solo de posición para comparación.

Este script es útil al depurar el contrato de landmarks base de 306 características.
El modelo de ejecución activo se entrena con `train_temporal_model.py`.
"""

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
from word_config import *

# Habilita el crecimiento dinámico de memoria de GPU para que TensorFlow no reserve toda la memoria de GPU de antemano.
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def data_augmentation(X_train, y_train):
    """Crea variantes conservadoras, solo de entrenamiento, de cada secuencia capturada.

    Los niveles de ruido son intencionalmente pequeños porque las características ya están normalizadas
    con respecto a la nariz; perturbaciones grandes crearían un movimiento de seña poco realista."""
    aug_sequences, aug_labels = [], []
    for seq, label in zip(X_train, y_train):
        # 1. La secuencia original se mantiene intacta.
        aug_sequences.append(seq)
        aug_labels.append(label)

        # 2. Ruido microscópico que simula un temblor milimétrico de la cámara.
        # Mantén la desviación baja porque las distancias relativas a la nariz son pequeñas.
        noise_1 = np.random.normal(0, 0.002, seq.shape)
        # Preserva los canales de visibilidad de la pose, asumiendo que los primeros 132 valores pertenecen a la pose.
        # Esto evita que el ruido corrompa los valores de visibilidad.
        aug_seq_1 = seq + noise_1
        aug_seq_1[:, 3::4] = seq[:, 3::4] # Restaura los valores de visibilidad originales.
        aug_sequences.append(aug_seq_1)
        aug_labels.append(label)

        # 3. Ruido ligero que simula la variación humana natural al señar.
        noise_2 = np.random.normal(0, 0.004, seq.shape)
        aug_seq_2 = seq + noise_2
        aug_seq_2[:, 3::4] = seq[:, 3::4] # Restaura los valores de visibilidad originales.
        aug_sequences.append(aug_seq_2)
        aug_labels.append(label)

    return aug_sequences, aug_labels

def load_raw_data_from_h5():
    """Carga secuencias solo de posición de longitud variable y etiquetas enteras desde archivos H5 por palabra."""
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

def build_model():
    """Construye la línea base recurrente solo de posición usada para comparar con las características temporales."""
    model = Sequential([

        # El masking le indica a las capas recurrentes que ignoren los fotogramas rellenados con ceros.
        Masking(mask_value=0.0, input_shape=(MAX_FRAMES, LENGTH_KEYPOINTS)),

        # SpatialDropout descarta canales de características completos, lo que desincentiva depender de una sola coordenada.
        SpatialDropout1D(0.2),

                # - LSTM bidireccional: lee la secuencia hacia adelante y hacia atrás, lo que ayuda a capturar
        # dependencias temporales en ambas direcciones. Cuando dos señas tienen inicios o finales similares,
        # el contexto completo de la secuencia ayuda a separarlas.
        # - L2: penaliza los pesos grandes, reduciendo el sobreajuste.
        # - Dropout: desactiva neuronas aleatorias durante el entrenamiento para que la red no dependa
        # demasiado de un pequeño subconjunto de características.

        Bidirectional(LSTM(64, return_sequences=True, activation='tanh', kernel_regularizer=l2(0.0005))),
        Dropout(0.3), # Un dropout más alto ayuda a combatir el sobreajuste.

        Bidirectional(LSTM(32, return_sequences=False, activation='tanh', kernel_regularizer=l2(0.0005))),
        Dropout(0.3), # Un dropout más alto ayuda a combatir el sobreajuste.

        # Capa densa de condensación.

        # - Dense ReLU: introduce no linealidad y condensa la información temporal
        # extraída por las LSTM en características lógicas más simples.
        # - Dense softmax: capa de salida multiclase que convierte los logits en probabilidades.

        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(len(WORDS), activation='softmax')
    ])

    # Reduce ligeramente el weight decay para equilibrar la penalización L2 añadida.
    optimizer = AdamW(learning_rate=0.0005, weight_decay=0.0005)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def plot_metrics(history, y_true, y_pred_classes):
    """Escribe las curvas de entrenamiento y una matriz de confusión normalizada para la revisión del experimento."""
    create_folder_if_not_exists(METRICS_FOLDER)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Accuracy')
    ax1.legend()

    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Loss')
    ax2.legend()
    plt.savefig(os.path.join(METRICS_FOLDER, 'training_history.png'))
    plt.close()

    cm = confusion_matrix(y_true, y_pred_classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=WORDS, yticklabels=WORDS)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Prediction')
    plt.savefig(os.path.join(METRICS_FOLDER, 'confusion_matrix.png'))
    plt.close()

if __name__ == "__main__":
    X_raw, y_raw = load_raw_data_from_h5()
    print(f"Total captured real samples: {len(X_raw)}")

    # División 80/20; mantén este experimento previo a los deltas sin conjunto de prueba hasta que cada palabra tenga suficientes muestras.
    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )

    X_train_aug, y_train_aug = data_augmentation(X_train_raw, y_train_raw)
    print(f"Training samples after augmentation: {len(X_train_aug)}")
    print(f"Validation samples (untouched): {len(X_val_raw)}")

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

    # - batch_size: número de videos vistos antes de cada actualización de gradiente; valores más grandes pueden
    # estabilizar los gradientes y usar mejor la GPU, pero requieren más memoria.
    # - epochs: más pasadas le dan al modelo más oportunidades de aprender, pero aumentan el riesgo de sobreajuste.
    # - EarlyStopping: detiene el entrenamiento automáticamente cuando la pérdida de validación deja de mejorar.

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=32,
        callbacks=[early_stop, checkpoint, reduce_lr]
    )

    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_val, axis=1)

    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_true, y_pred_classes, target_names=WORDS))

    plot_metrics(history, y_true, y_pred_classes)

# Notas de extensión futura del modelo:
# - Agregar bloques Conv1D antes de las capas recurrentes cuando el vocabulario crezca. Pueden
#   extraer patrones de movimiento de corto plazo, como una aceleración repentina de la mano o cierres
#   rápidos de los dedos, antes de que la LSTM modele la secuencia más larga.
# - Agregar LayerNormalization para modelos más grandes. Estabiliza las activaciones fotograma por fotograma
#   y ayuda a que redes más profundas se entrenen sin gradientes que explotan o se desvanecen.
# - Evaluar attention o self-attention cuando las señas incluyan segmentos neutros largos.
#   La atención puede aprender qué fotogramas portan la forma o el movimiento de mano más discriminativo.
# - Ampliar la aumentación de datos solo en los datos de entrenamiento. Las técnicas candidatas incluyen la
#   inversión temporal, el intercambio de manos y la generación sintética de secuencias; los conjuntos de validación y prueba
#   deben permanecer sin modificar para medir la generalización real.
# - Considerar conexiones residuales para pilas Conv1D/LSTM más profundas para que las capas posteriores puedan
#   seguir accediendo a la señal original de posición de los landmarks.

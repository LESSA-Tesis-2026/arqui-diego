"""Entrena el clasificador del alfabeto estático de LESSA a partir de keypoints H5 extraídos."""

import os
import h5py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from alphabet_config import *

# Habilita el crecimiento dinámico de memoria de GPU para que TensorFlow no reserve toda la memoria de GPU de antemano.
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(gpus)
    except RuntimeError as e:
        print(e)


def load_static_data():
    """Carga vectores estáticos del alfabeto de 306 valores y etiquetas enteras desde archivos H5 por letra."""
    X, y = [], []
    for label, letter in enumerate(ALPHABET):
        file_path = os.path.join(DATA_H5_FOLDER, f"{letter}.h5")
        if not os.path.exists(file_path): continue

        with h5py.File(file_path, 'r') as hf:
            for key in hf.keys():
                X.append(np.array(hf[key]))
                y.append(label)
    return np.array(X), np.array(y)

def build_static_model(input_dim, num_classes):
    """Construye el clasificador denso usado para las señas del alfabeto estático.

    Los tamaños de las capas forman un embudo desde las 306 características de landmarks hasta los logits del alfabeto,
    con normalización por lotes y dropout para reducir el sobreajuste."""
    model = Sequential([
        # La arquitectura de embudo comprime las 306 características de landmarks antes de la clasificación.
        Dense(256, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2), # Menor dropout a medida que la representación se estrecha.

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(num_classes, activation='softmax')
    ])

    # Reduce la tasa de aprendizaje para evitar saltos erráticos en la curva.
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def save_metrics(history, y_true, y_pred_classes):
    """Escribe las curvas de entrenamiento, la matriz de confusión y el reporte de clasificación en texto para su revisión."""
    create_folder_if_not_exists(METRICS_FOLDER)

    # 1. Gráfica del historial de entrenamiento.
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

    # 2. Matriz de confusión.
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=ALPHABET, yticklabels=ALPHABET)
    plt.title('Alphabet Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Prediction')
    plt.savefig(os.path.join(METRICS_FOLDER, 'confusion_matrix.png'))
    plt.close()

    # 3. Reporte en texto.
    report = classification_report(y_true, y_pred_classes, target_names=ALPHABET)
    with open(os.path.join(METRICS_FOLDER, 'classification_report.txt'), 'w') as f:
        f.write(report)

if __name__ == "__main__":
    X_raw, y_raw = load_static_data()
    print(f"Total loaded images: {len(X_raw)}")

    # División estricta en tres partes (70% entrenamiento, 15% validación, 15% prueba).
    X_train_raw, X_temp_raw, y_train_raw, y_temp_raw = train_test_split(
        X_raw, y_raw, test_size=0.30, random_state=42, stratify=y_raw
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp_raw, y_temp_raw, test_size=0.50, random_state=42, stratify=y_temp_raw
    )

    print(f"Train samples: {len(X_train_raw)} | Validation: {len(X_val)} | Test: {len(X_test)}")

    create_folder_if_not_exists(MODEL_FOLDER)
    model = build_static_model(LENGTH_KEYPOINTS, len(ALPHABET))
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
    ]

    # Un batch size de 64 promedia los gradientes y suaviza la curva de entrenamiento.
    history = model.fit(
        X_train_raw, y_train_raw,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=callbacks
    )

    print("\n--- EVALUATING MODEL ON UNSEEN TEST DATA ---")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Final real-world accuracy (Test Accuracy): {test_acc*100:.2f}%")

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    save_metrics(history, y_test, y_pred_classes)
    print(f"[SUCCESS] Model and metrics saved in folder '{MODEL_FOLDER}'")

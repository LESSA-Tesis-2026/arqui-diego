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
from config_alfabeto import *

# Habilitar memoria dinámica para la GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(gpus)
    except RuntimeError as e:
        print(e)


def load_static_data():
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
    model = Sequential([
        # Restauramos la "Ley del Embudo" para comprimir las 306 coordenadas limpiamente
        Dense(256, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2), # Reducimos ligeramente el castigo al estrechar la red
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(num_classes, activation='softmax')
    ])
    
    # Reducimos el paso de aprendizaje para evitar rebotes erráticos en la curva
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def save_metrics(history, y_true, y_pred_classes):
    create_folder_if_not_exists(METRICS_FOLDER)
    
    # 1. Gráfica de History
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

    # 2. Matriz de Confusión
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=ALPHABET, yticklabels=ALPHABET)
    plt.title('Matriz de Confusión - Alfabeto')
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción')
    plt.savefig(os.path.join(METRICS_FOLDER, 'confusion_matrix.png'))
    plt.close()
    
    # 3. Reporte en TXT
    report = classification_report(y_true, y_pred_classes, target_names=ALPHABET)
    with open(os.path.join(METRICS_FOLDER, 'classification_report.txt'), 'w') as f:
        f.write(report)

if __name__ == "__main__":
    X_raw, y_raw = load_static_data()
    print(f"Total de imágenes cargadas: {len(X_raw)}")
    
    # División estricta de 3 vías (70% Train, 15% Validation, 15% Test)
    X_train_raw, X_temp_raw, y_train_raw, y_temp_raw = train_test_split(
        X_raw, y_raw, test_size=0.30, random_state=42, stratify=y_raw
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp_raw, y_temp_raw, test_size=0.50, random_state=42, stratify=y_temp_raw
    )
    
    print(f"Muestras Train: {len(X_train_raw)} | Validation: {len(X_val)} | Test: {len(X_test)}")
    
    create_folder_if_not_exists(MODEL_FOLDER)
    model = build_static_model(LENGTH_KEYPOINTS, len(ALPHABET))
    model.summary()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
    ]
    
    # Aumentamos el batch_size a 64 para promediar los gradientes y suavizar la curva
    history = model.fit(
        X_train_raw, y_train_raw, 
        validation_data=(X_val, y_val), 
        epochs=100, 
        batch_size=64, 
        callbacks=callbacks
    )
    
    print("\n--- EVALUANDO MODELO CON DATOS DE TEST (VÍRGENES) ---")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Precisión final en el mundo real (Test Accuracy): {test_acc*100:.2f}%")
    
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    save_metrics(history, y_test, y_pred_classes)
    print(f"[ÉXITO] Modelo y métricas guardados en la carpeta '{MODEL_FOLDER}'")
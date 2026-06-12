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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from config_alfabeto import *

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
        Dense(256, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def save_metrics(history, y_true, y_pred_classes):
    create_folder_if_not_exists(METRICS_FOLDER)
    
    # 1. Gráfica de History
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Val')
    ax1.set_title('Precisión')
    ax1.legend()
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Val')
    ax2.set_title('Pérdida')
    ax2.legend()
    plt.savefig(os.path.join(METRICS_FOLDER, 'training_history.png'))
    plt.close()

    # 2. Matriz de Confusión
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=ALPHABET, yticklabels=ALPHABET)
    plt.title('Matriz de Confusión - Alfabeto')
    plt.savefig(os.path.join(METRICS_FOLDER, 'confusion_matrix.png'))
    plt.close()
    
    # 3. Reporte en TXT
    report = classification_report(y_true, y_pred_classes, target_names=ALPHABET)
    with open(os.path.join(METRICS_FOLDER, 'classification_report.txt'), 'w') as f:
        f.write(report)

if __name__ == "__main__":
    X, y = load_static_data()
    print(f"Total de imágenes cargadas: {len(X)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    create_folder_if_not_exists(MODEL_FOLDER)
    model = build_static_model(LENGTH_KEYPOINTS, len(ALPHABET))
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True)
    ]
    
    history = model.fit(X_train, y_train, validation_split=0.15, epochs=100, batch_size=32, callbacks=callbacks)
    
    print("\nEvaluando modelo...")
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    save_metrics(history, y_test, y_pred_classes)
    print(f"[ÉXITO] Modelo y métricas guardados en la carpeta '{MODEL_FOLDER}'")
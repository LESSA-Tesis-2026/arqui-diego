import os
import gc
import h5py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, SpatialDropout1D, Bidirectional, Conv1D, LayerNormalization
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l2
from config import *

# source ./venv_wsl/bin/activate

# Habilitar memoria dinámica para la GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(gpus)
    except RuntimeError as e:
        print(e)

def load_raw_data_from_h5():
    sequences, labels = [], []
    for label, word in enumerate(WORDS):
        file_path = os.path.join(DATA_PATH, f"{word}.h5")
        if not os.path.exists(file_path): continue
        
        with h5py.File(file_path, 'r') as hf:
            for key in hf.keys():
                seq = np.array(hf[key], dtype=np.float32)
                sequences.append(seq)
                labels.append(label)
                
    return sequences, labels

def data_augmentation(X_train, y_train):
    aug_sequences, aug_labels = [], []
    for seq, label in zip(X_train, y_train):
        seq = seq.astype(np.float32, copy=False)
        # 1. ORIGINAL
        aug_sequences.append(seq)
        aug_labels.append(label)
        
        # 2. RUIDO MICROSCÓPICO (Temblor milimétrico)
        noise_1 = np.random.normal(0, 0.002, seq.shape).astype(np.float32)
        aug_seq_1 = seq + noise_1
        aug_seq_1[:, 3::4] = seq[:, 3::4] # Protege la visibilidad
        aug_sequences.append(aug_seq_1)
        aug_labels.append(label)

        # 3. RUIDO LIGERO (Imperfección humana)
        noise_2 = np.random.normal(0, 0.004, seq.shape).astype(np.float32)
        aug_seq_2 = seq + noise_2
        aug_seq_2[:, 3::4] = seq[:, 3::4] # Protege la visibilidad
        aug_sequences.append(aug_seq_2)
        aug_labels.append(label)
        
    return aug_sequences, aug_labels

def compute_deltas(sequences):
    """
    Calcula la Velocidad (Delta) y Aceleración (Delta-Delta) para cada video.
    Transforma la entrada de 306 a 918 características.
    """
    processed_seqs = []
    for seq in sequences:
        seq = seq.astype(np.float32, copy=False)
        # Velocidad: Diferencia entre el frame actual y el anterior
        # Duplicamos el primer frame para no perder la longitud original
        delta = np.vstack([seq[0:1, :], np.diff(seq, axis=0).astype(np.float32)])
        
        # Aceleración: Diferencia entre la velocidad actual y la anterior
        delta_delta = np.vstack([delta[0:1, :], np.diff(delta, axis=0).astype(np.float32)])
        
        # Concatenamos Posición + Velocidad + Aceleración en el eje de las características
        combined_seq = np.concatenate([seq, delta, delta_delta], axis=-1).astype(np.float32)
        processed_seqs.append(combined_seq)
        
    return processed_seqs

def compute_deltas_and_pad(sequences, max_frames, batch_size=256):
    padded_batches = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start:start + batch_size]
        deltas = compute_deltas(batch)
        padded = pad_sequences(
            deltas,
            maxlen=max_frames,
            padding='post',
            truncating='post',
            dtype='float32'
        )
        padded_batches.append(padded)
    if not padded_batches:
        return np.empty((0, max_frames, 0), dtype='float32')
    return np.concatenate(padded_batches, axis=0)

def build_model(input_dim):
    model = Sequential([
        # 1. ENTRADA Y MÁSCARA
        Masking(mask_value=0.0, input_shape=(MAX_FRAMES, input_dim)),
        
        # 2. EXTRACTOR CONVOLUCIONAL (Nuevo)
        # Lee bloques de 3 frames para asimilar la posición, velocidad y aceleración
        Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
        LayerNormalization(), # Estabiliza las matemáticas después de la convolución
        
        SpatialDropout1D(0.2), 
        
        # 3. COMPRENSIÓN TEMPORAL
        Bidirectional(LSTM(128, return_sequences=True, activation='tanh', kernel_regularizer=l2(0.001))),
        Dropout(0.4),

        Bidirectional(LSTM(64, return_sequences=False, activation='tanh', kernel_regularizer=l2(0.001))),
        Dropout(0.4),
        
        # 4. CLASIFICADOR FINAL
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Dense(len(WORDS), activation='softmax')
    ])
    
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
    
    plt.figure(figsize=(14, 12)) # Aumentamos el tamaño para que quepan las 12 clases
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=WORDS, yticklabels=WORDS)
    plt.title('Matriz de Confusión Normalizada')
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción')
    plt.savefig(os.path.join(METRICS_FOLDER, 'confusion_matrix.png'))
    plt.close()

def save_hyperparameters(params):
    create_folder_if_not_exists(METRICS_FOLDER)
    file_path = os.path.join(METRICS_FOLDER, 'hyperparameters.txt')
    with open(file_path, 'w', encoding='utf-8') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

def save_final_metrics(text):
    create_folder_if_not_exists(METRICS_FOLDER)
    file_path = os.path.join(METRICS_FOLDER, 'final_metrics.txt')
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)

if __name__ == "__main__":
    # 1. Cargar datos
    X_raw, y_raw = load_raw_data_from_h5()
    print(f"Total de muestras reales capturadas: {len(X_raw)}")
    
    # 2. Dividir en Train (70%), Validation (15%) y Test (15%)
    # Primera división: Sacamos el 70% para Entrenamiento y dejamos 30% en un bloque temporal
    X_train_raw, X_temp_raw, y_train_raw, y_temp_raw = train_test_split(
        X_raw, y_raw, test_size=0.30, random_state=42, stratify=y_raw
    )
    
    # Segunda división: Partimos el bloque temporal a la mitad (15% y 15% del total original)
    X_val_raw, X_test_raw, y_val_raw, y_test_raw = train_test_split(
        X_temp_raw, y_temp_raw, test_size=0.50, random_state=42, stratify=y_temp_raw
    )
    
    print(f"Muestras Train: {len(X_train_raw)} | Validation: {len(X_val_raw)} | Test: {len(X_test_raw)}")
    
    # 3. Aumentar (OJO: El Data Augmentation SOLO se le hace al Train, jamás al Val o Test)
    X_train_aug, y_train_aug = data_augmentation(X_train_raw, y_train_raw)
    
    # 4. CALCULAR DELTAS Y DELTA-DELTAS
    #print("Calculando cinemática avanzada (Velocidad y Aceleración)...")
    #X_train_kinematics = compute_deltas(X_train_aug)
    #X_val_kinematics = compute_deltas(X_val_raw)
    #X_test_kinematics = compute_deltas(X_test_raw) # Nuevo: Calculamos deltas para el test
    
    #NUEVA_DIMENSION = X_train_kinematics[0].shape[-1]
    
    # 5. Padding
    #X_train = pad_sequences(X_train_kinematics, maxlen=MAX_FRAMES, padding='post', truncating='post', dtype='float32')
    #X_val = pad_sequences(X_val_kinematics, maxlen=MAX_FRAMES, padding='post', truncating='post', dtype='float32')
    #X_test = pad_sequences(X_test_kinematics, maxlen=MAX_FRAMES, padding='post', truncating='post', dtype='float32') # Nuevo
    
    print("Calculando cinemática avanzada (Velocidad y Aceleración)...")
    kinematics_batch_size = 64
    X_train = compute_deltas_and_pad(X_train_aug, MAX_FRAMES, batch_size=kinematics_batch_size)
    X_val = compute_deltas_and_pad(X_val_raw, MAX_FRAMES, batch_size=kinematics_batch_size)
    X_test = compute_deltas_and_pad(X_test_raw, MAX_FRAMES, batch_size=kinematics_batch_size) # Nuevo
    
    NUEVA_DIMENSION = X_train.shape[-1]

    # 6. Categorizar Etiquetas
    y_train = tf.keras.utils.to_categorical(y_train_aug, num_classes=len(WORDS))
    y_val = tf.keras.utils.to_categorical(y_val_raw, num_classes=len(WORDS))
    y_test = tf.keras.utils.to_categorical(y_test_raw, num_classes=len(WORDS)) # Nuevo

    del X_train_aug, y_train_aug, X_val_raw, X_test_raw, X_raw, y_raw
    gc.collect()

    # 7. Construir y compilar el modelo
    model = build_model(input_dim=NUEVA_DIMENSION)
    model.summary()

    create_folder_if_not_exists(MODEL_FOLDER_PATH)
    
    # 8. Callbacks
    epochs = 300
    batch_size = 64
    early_stop_patience = 25
    reduce_lr_factor = 0.5
    reduce_lr_patience = 10
    reduce_lr_min_lr = 0.00001
    lstm_units = [128, 64]

    early_stop = EarlyStopping(monitor='val_loss', patience=early_stop_patience, restore_best_weights=True)
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        min_lr=reduce_lr_min_lr,
        verbose=1
    )

    save_hyperparameters({
        'epochs': epochs,
        'batch_size': batch_size,
        'early_stopping_patience': early_stop_patience,
        'reduce_lr_factor': reduce_lr_factor,
        'reduce_lr_patience': reduce_lr_patience,
        'reduce_lr_min_lr': reduce_lr_min_lr,
        'lstm_units': lstm_units
    })

    # 9. Entrenamiento
    # Aumentamos el batch_size a 32 para estabilizar las 12 clases
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, checkpoint, reduce_lr]
    )

    # 10. Evaluación con el set de TEST (Datos Vírgenes)
    print("\n--- EVALUANDO MODELO CON DATOS DE TEST ---")
    
    # Evaluamos la pérdida y precisión exactas
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    final_acc_text = f"Precisión final en el mundo real (Test Accuracy): {test_acc*100:.2f}%"
    print(final_acc_text)
    
    # Predicciones para el reporte y matriz
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1) # y_true ahora viene del Test

    print("\n--- REPORTE DE CLASIFICACIÓN (SET DE TEST) ---")
    classification_text = classification_report(y_true, y_pred_classes, target_names=WORDS)
    print(classification_text)

    plot_metrics(history, y_true, y_pred_classes)

    final_metrics_text = "\n".join([
        "--- EVALUANDO MODELO CON DATOS DE TEST ---",
        final_acc_text,
        "",
        "--- REPORTE DE CLASIFICACION (SET DE TEST) ---",
        classification_text
    ])
    save_final_metrics(final_metrics_text)
"""Train the active LESSA word/phrase model with temporal features.

The model receives position + velocity + acceleration features per frame. This
is the training counterpart of the API's `(60, 918)` word-model input contract.
"""

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
from word_config import *

# Enable dynamic GPU memory growth so TensorFlow does not reserve all GPU memory up front.
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(gpus)
    except RuntimeError as e:
        print(e)

def load_raw_data_from_h5():
    """Load variable-length word/phrase sequences and integer labels from per-word H5 files."""
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
    """Create conservative training-only variants of each captured sequence.

    Visibility channels are restored after adding noise so MediaPipe confidence values
    remain valid while position coordinates receive small perturbations."""
    aug_sequences, aug_labels = [], []
    for seq, label in zip(X_train, y_train):
        seq = seq.astype(np.float32, copy=False)
        # 1. Original sequence kept intact.
        aug_sequences.append(seq)
        aug_labels.append(label)

        # 2. Microscopic noise that simulates millimetric camera shake.
        noise_1 = np.random.normal(0, 0.002, seq.shape).astype(np.float32)
        aug_seq_1 = seq + noise_1
        aug_seq_1[:, 3::4] = seq[:, 3::4] # Preserve MediaPipe visibility values.
        aug_sequences.append(aug_seq_1)
        aug_labels.append(label)

        # 3. Light noise that simulates small human signing variation.
        noise_2 = np.random.normal(0, 0.004, seq.shape).astype(np.float32)
        aug_seq_2 = seq + noise_2
        aug_seq_2[:, 3::4] = seq[:, 3::4] # Preserve MediaPipe visibility values.
        aug_sequences.append(aug_seq_2)
        aug_labels.append(label)

    return aug_sequences, aug_labels

def compute_deltas(sequences):
    """
    Compute velocity (delta) and acceleration (delta-delta) for each video.
    This transforms each frame from 306 position features to 918 kinematic features.
    """
    processed_seqs = []
    for seq in sequences:
        seq = seq.astype(np.float32, copy=False)
        # Velocity: difference between the current and previous frame.
        # The first frame is duplicated so sequence length is preserved.
        delta = np.vstack([seq[0:1, :], np.diff(seq, axis=0).astype(np.float32)])

        # Acceleration: difference between the current and previous velocity vectors.
        delta_delta = np.vstack([delta[0:1, :], np.diff(delta, axis=0).astype(np.float32)])

        # Concatenate position + velocity + acceleration along the feature axis.
        combined_seq = np.concatenate([seq, delta, delta_delta], axis=-1).astype(np.float32)
        processed_seqs.append(combined_seq)

    return processed_seqs

def compute_deltas_and_pad(sequences, max_frames, batch_size=256):
    """Convert sequences to temporal features in batches, then pad/truncate to model length.

    Batching avoids holding all intermediate 918-feature sequences in memory at once,
    which matters when the dataset grows."""
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
    """Build the active temporal word/phrase classifier.

    Conv1D extracts short motion patterns, bidirectional LSTMs model sequence context,
    and dense layers classify the final sign label."""
    model = Sequential([
        # 1. Input and masking layer.
        Masking(mask_value=0.0, input_shape=(MAX_FRAMES, input_dim)),

        # 2. Convolutional feature extractor.
        # Read three-frame blocks to learn position, velocity, and acceleration patterns.
        Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
        LayerNormalization(), # Stabilize activations after the convolution.

        SpatialDropout1D(0.2),

        # 3. Temporal understanding.
        Bidirectional(LSTM(128, return_sequences=True, activation='tanh', kernel_regularizer=l2(0.001))),
        Dropout(0.4),

        Bidirectional(LSTM(64, return_sequences=False, activation='tanh', kernel_regularizer=l2(0.001))),
        Dropout(0.4),

        # 4. Final classifier.
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
    """Write training curves and a normalized confusion matrix for the held-out test split."""
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

    plt.figure(figsize=(14, 12)) # Increase figure size so all 12 classes remain readable.
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=WORDS, yticklabels=WORDS)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Prediction')
    plt.savefig(os.path.join(METRICS_FOLDER, 'confusion_matrix.png'))
    plt.close()

def save_hyperparameters(params):
    """Persist run settings beside metrics so future developers can reproduce the training run."""
    create_folder_if_not_exists(METRICS_FOLDER)
    file_path = os.path.join(METRICS_FOLDER, 'hyperparameters.txt')
    with open(file_path, 'w', encoding='utf-8') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

def save_final_metrics(text):
    """Persist the final test accuracy and classification report beside generated charts."""
    create_folder_if_not_exists(METRICS_FOLDER)
    file_path = os.path.join(METRICS_FOLDER, 'final_metrics.txt')
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)

if __name__ == "__main__":
    # 1. Load data.
    X_raw, y_raw = load_raw_data_from_h5()
    print(f"Total captured real samples: {len(X_raw)}")

    # 2. Split into train (70%), validation (15%), and test (15%).
    # First split: keep 70% for training and leave 30% in a temporary holdout block.
    X_train_raw, X_temp_raw, y_train_raw, y_temp_raw = train_test_split(
        X_raw, y_raw, test_size=0.30, random_state=42, stratify=y_raw
    )

    # Second split: divide the temporary block in half (15% and 15% of the original dataset).
    X_val_raw, X_test_raw, y_val_raw, y_test_raw = train_test_split(
        X_temp_raw, y_temp_raw, test_size=0.50, random_state=42, stratify=y_temp_raw
    )

    print(f"Train samples: {len(X_train_raw)} | Validation: {len(X_val_raw)} | Test: {len(X_test_raw)}")

    # 3. Augment training data only; never augment validation or test data.
    X_train_aug, y_train_aug = data_augmentation(X_train_raw, y_train_raw)

    # 4. Build the model-ready temporal tensors.
    # `compute_deltas_and_pad` creates position + velocity + acceleration features and
    # applies the same 60-frame padding contract used by the runtime API.
    print("Computing advanced kinematics (velocity and acceleration)...")
    kinematics_batch_size = 64
    X_train = compute_deltas_and_pad(X_train_aug, MAX_FRAMES, batch_size=kinematics_batch_size)
    X_val = compute_deltas_and_pad(X_val_raw, MAX_FRAMES, batch_size=kinematics_batch_size)
    X_test = compute_deltas_and_pad(X_test_raw, MAX_FRAMES, batch_size=kinematics_batch_size) # Same transformation for the untouched test set.

    input_dimension = X_train.shape[-1]

    # 5. One-hot encode labels.
    y_train = tf.keras.utils.to_categorical(y_train_aug, num_classes=len(WORDS))
    y_val = tf.keras.utils.to_categorical(y_val_raw, num_classes=len(WORDS))
    y_test = tf.keras.utils.to_categorical(y_test_raw, num_classes=len(WORDS))

    del X_train_aug, y_train_aug, X_val_raw, X_test_raw, X_raw, y_raw
    gc.collect()

    # 6. Build and compile the model.
    model = build_model(input_dim=input_dimension)
    model.summary()

    create_folder_if_not_exists(MODEL_FOLDER_PATH)

    # 7. Training callbacks.
    epochs = 300
    batch_size = 64
    early_stop_patience = 25
    reduce_lr_factor = 0.5
    reduce_lr_patience = 10
    reduce_lr_min_lr = 0.00001
    lstm_units = [256, 128]

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

    # 8. Training.
    # Batch size is tuned to stabilize training across the 12 word classes.
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, checkpoint, reduce_lr]
    )

    # 9. Evaluate on the untouched test set.
    print("\n--- EVALUATING MODEL ON TEST DATA ---")

    # Evaluate the exact loss and accuracy.
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    final_acc_text = f"Final real-world accuracy (Test Accuracy): {test_acc*100:.2f}%"
    print(final_acc_text)

    # Predictions used for the report and confusion matrix.
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1) # y_true now comes from the test set.

    print("\n--- CLASSIFICATION REPORT (TEST SET) ---")
    classification_text = classification_report(y_true, y_pred_classes, target_names=WORDS)
    print(classification_text)

    plot_metrics(history, y_true, y_pred_classes)

    final_metrics_text = "\n".join([
        "--- EVALUATING MODEL ON TEST DATA ---",
        final_acc_text,
        "",
        "--- CLASSIFICATION REPORT (TEST SET) ---",
        classification_text
    ])
    save_final_metrics(final_metrics_text)

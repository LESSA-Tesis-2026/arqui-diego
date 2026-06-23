"""Train a position-only word/phrase experiment for comparison.

This script is useful when debugging the baseline 306-feature landmark contract.
The active runtime model is trained by `train_temporal_model.py`.
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

# Enable dynamic GPU memory growth so TensorFlow does not reserve all GPU memory up front.
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def data_augmentation(X_train, y_train):
    """Create conservative training-only variants of each captured sequence.

    The noise levels are intentionally small because features are already normalized
    relative to the nose; large perturbations would create unrealistic signing motion."""
    aug_sequences, aug_labels = [], []
    for seq, label in zip(X_train, y_train):
        # 1. Original sequence kept intact.
        aug_sequences.append(seq)
        aug_labels.append(label)

        # 2. Microscopic noise that simulates millimetric camera shake.
        # Keep the deviation low because nose-relative distances are small.
        noise_1 = np.random.normal(0, 0.002, seq.shape)
        # Preserve pose visibility channels, assuming the first 132 values belong to pose.
        # This prevents noise from corrupting visibility values.
        aug_seq_1 = seq + noise_1
        aug_seq_1[:, 3::4] = seq[:, 3::4] # Restore original visibility values.
        aug_sequences.append(aug_seq_1)
        aug_labels.append(label)

        # 3. Light noise that simulates natural human signing variation.
        noise_2 = np.random.normal(0, 0.004, seq.shape)
        aug_seq_2 = seq + noise_2
        aug_seq_2[:, 3::4] = seq[:, 3::4] # Restore original visibility values.
        aug_sequences.append(aug_seq_2)
        aug_labels.append(label)

    return aug_sequences, aug_labels

def load_raw_data_from_h5():
    """Load variable-length position-only sequences and integer labels from per-word H5 files."""
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
    """Build the position-only recurrent baseline used for comparison with temporal features."""
    model = Sequential([

        # Masking tells the recurrent layers to ignore zero-padded frames.
        Masking(mask_value=0.0, input_shape=(MAX_FRAMES, LENGTH_KEYPOINTS)),

        # SpatialDropout drops whole feature channels, which discourages reliance on one coordinate.
        SpatialDropout1D(0.2),

                # - Bidirectional LSTM: reads the sequence forward and backward, which helps capture
        # temporal dependencies in both directions. When two signs have similar starts or endings,
        # the complete sequence context helps separate them.
        # - L2: penalizes large weights, reducing overfitting.
        # - Dropout: disables random neurons during training so the network does not depend
        # too strongly on a small subset of features.

        Bidirectional(LSTM(64, return_sequences=True, activation='tanh', kernel_regularizer=l2(0.0005))),
        Dropout(0.3), # Higher dropout helps fight overfitting.

        Bidirectional(LSTM(32, return_sequences=False, activation='tanh', kernel_regularizer=l2(0.0005))),
        Dropout(0.3), # Higher dropout helps fight overfitting.

        # Dense condensation layer.

        # - Dense ReLU: introduces non-linearity and condenses the temporal information
        # extracted by the LSTMs into simpler logical features.
        # - Dense softmax: multiclass output layer that converts logits into probabilities.

        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(len(WORDS), activation='softmax')
    ])

    # Slightly reduce weight decay to balance the added L2 penalty.
    optimizer = AdamW(learning_rate=0.0005, weight_decay=0.0005)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def plot_metrics(history, y_true, y_pred_classes):
    """Write training curves and a normalized confusion matrix for experiment review."""
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

    # 80/20 split; keep this pre-delta experiment without a test set until each word has enough samples.
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

    # - batch_size: number of videos seen before each gradient update; larger values can
    # stabilize gradients and use the GPU better, but require more memory.
    # - epochs: more passes give the model more learning chances but increase overfitting risk.
    # - EarlyStopping: stops training automatically when validation loss stops improving.

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

# Future model-extension notes:
# - Add Conv1D blocks before the recurrent layers when the vocabulary grows. They can
#   extract short-term motion patterns, such as sudden hand acceleration or quick
#   finger closures, before the LSTM models the longer sequence.
# - Add LayerNormalization for larger models. It stabilizes activations frame by frame
#   and helps deeper networks train without exploding or vanishing gradients.
# - Evaluate attention or self-attention when signs include long neutral segments.
#   Attention can learn which frames carry the most discriminative hand shape or motion.
# - Expand data augmentation only on training data. Candidate techniques include time
#   reversal, hand swapping, and synthetic sequence generation; validation and test sets
#   must stay untouched to measure real generalization.
# - Consider residual connections for deeper Conv1D/LSTM stacks so later layers can
#   still access the original landmark-position signal.

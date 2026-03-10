import numpy as np
from model import get_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from helpers import get_word_ids, get_sequences_and_labels
from constants import *

def training_model(model_path, epochs=500):
    word_ids = get_word_ids(WORDS_JSON_PATH ) # ['word1', 'word2', 'word3]
    
    sequences, labels = get_sequences_and_labels(word_ids)
    
    sequences = pad_sequences(sequences, maxlen=int(MODEL_FRAMES), padding='pre', truncating='post', dtype='float32')
    
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    
    # Normalizar los datos para mejor convergencia
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X = (X - X_mean) / (X_std + 1e-8)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Configurar early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Cambiar de 'accuracy' a 'val_loss'
        patience=30,  # Aumentar de 10 a 30
        restore_best_weights=True,
        verbose=1
    )
    
    # Agregar reducción de learning rate
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=0.00001,
        verbose=1
    )
    
    model = get_model(int(MODEL_FRAMES), len(word_ids))
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=16, callbacks=[early_stopping, reduce_lr])
    
    model.summary()
    model.save(model_path)

if __name__ == "__main__":
    training_model(MODEL_PATH)
    
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from constants import LENGTH_KEYPOINTS

def get_model(max_length_frames, output_length: int):
    model = Sequential()
    
    # Primera capa LSTM con más unidades
    model.add(LSTM(128, return_sequences=True, input_shape=(max_length_frames, LENGTH_KEYPOINTS)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Segunda capa LSTM
    model.add(LSTM(128, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Tercera capa LSTM
    model.add(LSTM(64, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Capas densas
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(output_length, activation='softmax'))
    
    # Optimizador con learning rate personalizado
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
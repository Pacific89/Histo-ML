import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation

def _mlp_regressor(X, y, epochs=500, batch_size=64):

    model = Sequential([
    Flatten(input_shape=(512,)),
    Dense(100, activation='relu', kernel_initializer='he_normal'),
    Dense(50, activation='relu', kernel_initializer='he_normal'),
    Dense(10, activation='relu', kernel_initializer='he_normal'),
    Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.summary()
    model.fit(X, y, epochs=epochs, batch_size=batch_size)


def _mlp_classifier(X, y):
    num_targets = len(set(y))

    model = Sequential([
    Flatten(input_shape=(512,)),
    Dense(100, activation='relu'),
    Dense(50, activation='relu'),  
    Dense(10, activation='relu'), 
    Dense(num_targets, activation='softmax'), 
    ])

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    model.summary()
    model.fit(X, y, epochs=500, 
          batch_size=200, 
          validation_split=0.2)
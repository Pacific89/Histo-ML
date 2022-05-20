import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
import tensorboard


def get_tensorboard_callback(log_dir="./logs"):

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    return tensorboard_callback

def _mlp_regressor(X, y, epochs=500, batch_size=200, validation_split=0.2):

    model = Sequential([
    Flatten(input_shape=(512,)),
    Dense(100, activation='relu', kernel_initializer='he_normal'),
    Dense(50, activation='relu', kernel_initializer='he_normal'),
    Dense(10, activation='relu', kernel_initializer='he_normal'),
    Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.summary()

    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split,  callbacks=[get_tensorboard_callback()])


def _mlp_classifier(X, y, epochs=500, batch_size=200, validation_split=0.2):

    model = Sequential([
    Flatten(input_shape=(512,)),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),  
    Dense(128, activation='relu'),
    Dense(64, activation='relu'), 
    Dense(32, activation='relu'), 
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(2, activation='softmax'), 
    ])

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    model.summary()
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[get_tensorboard_callback()])
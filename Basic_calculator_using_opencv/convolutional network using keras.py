import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Activation, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import copy
import random
import numpy as np


#Load the data of mnist dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Explore the data
#plt.imshow(X_train[0], cmap='gray')
#plt.show()

#Reshape to fit the model
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

#One hot encode the labels of the image
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#Create the model
model = Sequential()

model.add(
    Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1))
)
model.add(
    Conv2D(32, (3, 3), activation='relu')
)
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath="calculator_weights.hdf5", verbose=2, save_best_only=True)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=10, callbacks=[checkpointer])

model_json = model.to_json()

with open("Calculator_model", 'w') as json_file:
    json_file.write(model_json)




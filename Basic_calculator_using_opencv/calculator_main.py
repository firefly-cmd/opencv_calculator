import matplotlib.pyplot as plt
import copy
import random
import numpy as np
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Activation, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential

###LOAD THE MODEL###

with open("Calculator_model", 'r') as json_file:
    json_SavedModel = json_file.read()

model = tf.keras.models.model_from_json(json_SavedModel)
model.load_weights("calculator_weights.hdf5")
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
#model.summary()












###DRAW A NUMBER TO THE BOARD###

drawing = False
ex = -1
ey = -1


def draw_line(event, x, y, flags, params):
    global ex, ey, exlast, eylast, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ex, ey = x, y


    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(img, (ex, ey), (x, y), (255, 0, 0), thickness=20)
            ex, ey = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (ex, ey), (x, y), (255, 0, 0), thickness=20)


img = np.zeros((512, 512, 3), np.int8)

cv2.namedWindow('my_drawing')

cv2.setMouseCallback('my_drawing', draw_line)


while True:
    cv2.imshow('my_drawing', img)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cv2.destroyAllWindows()


###END  OF DRAWING NUMBER####


##PREDICT THE NUMBER WRITEN ON THE SCREEN###
predict_image = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)
predict_image = np.expand_dims(np.expand_dims(cv2.resize(predict_image, (28, 28)), axis=2), axis=0)
prediction = model.predict(predict_image)
print(prediction)
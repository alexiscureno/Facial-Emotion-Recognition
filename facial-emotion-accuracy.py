import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.layers import BatchNormalization
from tensorflow import keras

import argparse
import random
import time


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mode = "display"


#modelo 1
'''
# Create the model
model = Sequential()

#First Layer (lleva inputs)
model.add(Conv2D(48, (3, 3), padding = 'same', activation = 'relu', input_shape = (48, 48, 3)))
model.add(BatchNormalization())
model.add(Conv2D(48, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

#Flattening
model.add(Flatten())

model.add(Dense(128, activation='softmax'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

'''

#modelo 2
model_1 = Sequential()

model_1.add(Conv2D(64,(3,3),padding="same",activation='relu',input_shape=(48,48,1)))
model_1.add(Conv2D(64,(3,3),padding="same",activation='relu'))
model_1.add(MaxPooling2D(pool_size=(2,2)))
model_1.add(Dropout(0.5))

model_1.add(Conv2D(128,(3,3),activation='relu'))
model_1.add(Conv2D(128,(3,3),activation='relu'))
model_1.add(MaxPooling2D(pool_size=(2,2)))
model_1.add(Dropout(0.5))

model_1.add(Conv2D(256,(3,3),activation='relu'))
model_1.add(Conv2D(256,(3,3),activation='relu'))
model_1.add(Conv2D(256,(3,3),activation='relu'))
model_1.add(MaxPooling2D(pool_size=(2,2)))
model_1.add(Dropout(0.5))

model_1.add(Flatten())
model_1.add(Dense(128,activation='relu'))
model_1.add(Dense(64,activation='relu'))
model_1.add(Dense(32,activation='relu'))
model_1.add(Dropout(0.6))
model_1.add(Dense(7,activation='softmax'))


#compiling model
model_1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#print(model_1.summary())


import tensorflow as tf
print(keras.__version__)
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def emotion_detect(frame):
    model_1.load_weights('model_v2.h5')

    emotions_dictionary = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sadness", 6: "Surprise"}
    #{0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    faceclassifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceclassifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 255), 3)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model_1.predict(cropped_img)
        max_index = int(np.argmax(prediction))
        cv2.putText(frame, emotions_dictionary[max_index], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                    cv2.LINE_AA)
    return frame

# Open CV Webcam

capture = cv2.VideoCapture(0)
while(True):
    _, fram = capture.read()
    img_output = emotion_detect(fram)
    cv2.imshow('Alexis Cam', img_output)
    if cv2.waitKey(1) == ord('q'):
        break
#When done, release the capture
capture.release()
cv2.destroyAllWindows()

# Open CV Webcam 2
'''
size = 7
webcam = cv2.VideoCapture(0)  # Use camera 0
emotions_dictionary = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sadness", 6: "Surprise"}
# We load the xml file
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model_1.load_weights('model_v2.h5')

while True:
    (rval, im) = webcam.read()
    im = cv2.flip(im, 1, 1)  # Flip to act as a mirror
    im_res = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

    # Resize the image to speed up detection
    #mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

    # detect MultiScale / faces
    faces = classifier.detectMultiScale(im_res)

    # Draw rectangles around each face
    for f in faces:
        (x, y, w, h) = [v * size for v in f]  # Scale the shapesize backup
        # Save just the rectangle faces in SubRecFaces
        face_img = im[y:y + h, x:x + w]
        resized = cv2.resize(face_img, (48, 48))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 48, 48, 1))
        reshaped = np.vstack([reshaped])
        result = model_1.predict(reshaped)
        #answer = model_1.predict_classes(test_image)

        # print(result)

        label = np.argmax(result, axis=1)[0]

        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 225), 2)
        cv2.rectangle(im, (x, y - 40), (x + w, y), (0, 0, 225), -1)
        cv2.putText(im, emotions_dictionary[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show the image
    cv2.imshow('Alexis Cam', im)
    key = cv2.waitKey(10)
    # if Esc key is press then break out of the loop
    if key == 27:  # The Esc key
        break
# Stop video
webcam.release()

# Close all started windows
cv2.destroyAllWindows()
'''



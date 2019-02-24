from sklearn.model_selection import train_test_split
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.utils import np_utils
import cv2
import os
from keras import backend as K

K.set_image_dim_ordering('th')

# SIZE OF IMAGES
H = 50
W = 50


# ------LOAD X,Y----------


def load_pos(folder):  # sends +folder path and returns their binary images and class
    images = []
    out = []
    for filename in os.listdir(folder):
        img1 =(cv2.imread(os.path.join(folder, filename), 0))
        if img1 is not None:
            img = cv2.resize(img1, (H, W))
            images.append(img)
            out.append(1)
    return (np.array(images), np.array(out))


# ---------NEgative examples---------

def load_neg(folder):  # sends -folder path and returns their binary images and class
    images = []
    out = []
    for filename in os.listdir(folder):
        img1 = cv2.imread(os.path.join(folder, filename), 0)
        if img1 is not None:
            img = cv2.resize(img1, (H, W))
            images.append(img)
            out.append(0)
    return (np.array(images), np.array(out))


# -------------------------------------
# COOL IDEA
ls = ["flat", "half", "quarter", "sharp", "whole", "negative"]
#-----------


for IT in range (len(ls)-1):

    str = ls[IT]

    X, Y = load_pos(str)

    for YT in range (len(ls)):
        if (YT != IT):
            x1, y1 = load_neg(ls[YT])

            X = np.append(X, x1, axis=0)
            Y = np.append(Y, y1, axis=0)


    #------------------------
    # ------TRAIN TEST SPLIT----------


    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=42)

    # --------------------------

    # Reshaping for use in CNN as [samples][height][weight][channels] channels = 1 for grey scale

    X_train = X_train.reshape(X_train.shape[0], H, W, 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], H, W, 1).astype('float32')

    y_train = y_train.reshape(y_train.shape[0], 1).astype('float32')        #   IN BETA
    y_test = y_test.reshape(y_test.shape[0], 1).astype('float32')


    # Normalizing intensity values from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255


    # -------------Defining Model------------


    def baseline_model():
        # Using Sequential models
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(5, 5), data_format='channels_last', input_shape=(H, W, 1), activation='relu',
               strides=1)) 
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=32, kernel_size=(5, 5), data_format='channels_last', input_shape=(H, W, 1),
                         activation='relu',
                         strides=1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        # Compiling Model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model


    # Building model
    model = baseline_model()

    # Training the model
    model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), batch_size=200, verbose=0)

    # Evaluating
    score = model.evaluate(X_test, y_test, verbose=0)

    # Printing error
    print("CNN Error: %.2f%%" % (100 - score[1] * 100))


    # SAVING THE WEIGHTS
    str+=".h5"
    model.save_weights(str)

# yy = model.predict(X_train)
# yy[yy>=0.5] = 1
# yy[yy<0.5] = 0
# for i in range(y_train.shape[0]):
#     print (yy[i][0], y_train[i][0])



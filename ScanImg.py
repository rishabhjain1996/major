import numpy as np
from keras import backend as K
import cv2
K.set_image_dim_ordering('th')
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential

def Scan (str,img, fx, fy):
    # -------------Redefining Model------------

    H = 50
    W = 50

    def baseline_model():
        # Using Sequential models
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(5, 5), data_format='channels_last', input_shape=(H, W, 1),
                         activation='relu',
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

    # Loading weigths
    str += ".h5"
    model.load_weights(str, by_name=False)

    #----------------------------------------------

    stridex=int(fx/2)
    stridey=int(fy/3)

    result = np.zeros((img.shape[0]-fx+1,img.shape[1]-fy+1))
    for i in range (0,img.shape[0]-fx+1,stridex):
        for j in range (0,img.shape[1]-fy+1,stridey):
            ex = i+fx-1
            ey = j+fy-1

            crop = img[i:i+fx,j:j+fy]


            crop = cv2.resize(crop, (H, W))
            x = crop.reshape(1, H, W, 1).astype('float32')

            x = x / 255

            y = model.predict(x)
            if (str == "half"):
                y[y >= 0.7] = 1
                y[y < 0.7] = 0
            elif (str == "whole"):
                y[y >= 0.4] = 1
                y[y < 0.4] = 0
            y[y >= 0.5] = 1
            y[y < 0.5] = 0

            result[i][j] = y[0][0]

    return result
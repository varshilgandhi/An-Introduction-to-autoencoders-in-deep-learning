# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 04:16:04 2021

@author: abc
"""

from matplotlib.pyplot import imshow
import numpy as np
import cv2

from keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential

np.random.seed(42)

SIZE = 256

#let's start importing images

#create empty array
img_data = []

#read the image
img = cv2.imread('monalisa.jpg',1)  #1 means importing as color image
                                    #0 means importing as gray scale image

#opencv imports images as BGR not as RGB
#so convert BGR TO RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#RESIZE OUR IMAGE INTO 256
img = cv2.resize(img, (SIZE, SIZE))

#append image into array
img_data.append(img_to_array(img))

#Reshape our array so we can easily apply many input images[if needed]
img_array = np.reshape(img_data, (len(img_data), SIZE, SIZE, 3))

#Normalize our image with divide by 255 so that values range between 0 and 1
img_array = img_array.astype('float32') / 255.


#encoding
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(SIZE, SIZE, 3)))
model.add(MaxPooling2D((2,2), padding='same'))
model.add(Conv2D(8, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2), padding='same'))
model.add(Conv2D(8, (3,3), activation='relu', padding='same'))

model.add(MaxPooling2D((2,2), padding='same'))

#decoding
model.add(Conv2D(8, (3,3), activation='relu', padding = 'same'))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(8, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(3, (3,3), activation='relu', padding='same'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.summary()


#fit our model
model.fit(img_array, img_array, epochs=5, shuffle=True)

#predict the model
pred = model.predict(img_array)

#let's see our prediction
imshow(pred[0].reshape(SIZE, SIZE, 3))




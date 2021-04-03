#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import cv2
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_test.shape

plt.imshow(x_train[2])
plt.show()
plt.imshow(x_train[2], cmap = plt.cm.binary)

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)
plt.imshow(x_train[2], cmap = plt.cm.binary)

print(x_train[0])
print(y_train[0])

IMG_SIZE = 28
x_trainr = np.array(x_train).reshape(-1,IMG_SIZE,IMG_SIZE,1)
x_testr = np.array(x_test).reshape(-1,IMG_SIZE,IMG_SIZE,1)
print("Training samples dimension: ", x_trainr.shape)
print("Test samples dimension: ", x_testr.shape)

model = Sequential()

model.add(Conv2D(64,(3,3),input_shape = x_trainr.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(32))
model.add(Activation("relu"))

model.add(Dense(10))
model.add(Activation("softmax"))

model.summary()

print("Total training Samples = ", len(x_trainr))


model.compile(loss="sparse_categorical_crossentropy",optimizer = "adam", metrics = ['accuracy'])
model.fit(x_trainr, y_train,epochs=5, validation_split = 0.3)

test_loss, test_acc = model.evaluate(x_testr,y_test)
print("Test loss on 10,000 test samples", test_loss)
print("validation accuracy on 10,000 test samples", test_acc)

predictions = model.predict([x_testr])
print(predictions)

print(np.argmax(predictions[0]))

plt.imshow(x_test[0])

from skimage import io
img =io.imread('C:\\Users\\87gou\\Desktop\\Handwritten\\five.png')

gray =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray.shape

resized = cv2.resize(gray, (28,28), interpolation=cv2.INTER_AREA)

resized.shape

newimg = tf.keras.utils.normalize (resized, axis = 1)

newimg = np.array(newimg).reshape(-1, IMG_SIZE, IMG_SIZE,1)

newimg.shape

predictions = model.predict(newimg)

predictions = model.predict(newimg)

print(np.argmax(predictions))


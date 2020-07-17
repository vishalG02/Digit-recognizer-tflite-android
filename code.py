# -*- coding: utf-8 -*-


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt



mnist = tf.keras.datasets.mnist
(images_train, labels_train),(images_test, labels_test) = mnist.load_data()

class_names = ["zero","one","two","three","four","five","six","seven","eight","nine"]

plt.figure()
plt.imshow(images_train[0])
plt.colorbar()
plt.grid(False)
plt.xlabel("Classification label: {}".format(labels_train[0]))
plt.show()
# normalize the image data  255 pixels
images_train = images_train / 255.0
images_test = images_test / 255.0
#image is 2d array
#convert 2d image data array into 1d array using flatten
#dropout is used to avoid overfitting
# 2 hidden layers in network 512neurons
# output layers 10 neurons since weare classifying 0 to 9
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28,28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#train model
model.fit(images_train, labels_train, epochs=16)
#evaluate using test data
test_loss, test_acc = model.evaluate(images_test, labels_test)
print('Test accuracy:', test_acc)


#save model
kearas_file = "digit2.h5"
tf.keras.models.save_model(model,kearas_file)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tfmodel = converter.convert()
open("digit2.tflite","wb").write(tfmodel)

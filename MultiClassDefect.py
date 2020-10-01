# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
import glob
from tensorflow.keras.preprocessing.image import load_img, img_to_array


train = ImageDataGenerator(rescale = 1/255)

train_dataset = train.flow_from_directory('input/Defect/train',
                                          target_size=(224,224), batch_size=3, class_mode='categorical')

test_dataset = train.flow_from_directory('input/Defect/validation',
                                          target_size=(224,224), batch_size=3, class_mode='categorical')



print(train_dataset.classes)


from keras.models import *
from keras.layers import *

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.MaxPool2D(2, 2),
                                    tf.keras.layers.Dropout(0.2),

                                    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.MaxPool2D(2, 2),
                                    tf.keras.layers.Dropout(0.2),

                                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.MaxPool2D(2, 2),
                                    tf.keras.layers.Dropout(0.2),

                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation='relu'),
                                    tf.keras.layers.Dense(units=128, activation='relu'),
                                    tf.keras.layers.Dense(6, activation='softmax')
                                    ])

model.compile(loss='categorical_crossentropy', optimizer = RMSprop(lr=0.001), metrics=['categorical_accuracy'])

print("Model training started...")


model_fit = model.fit(train_dataset, epochs=20, batch_size=16,validation_data = test_dataset)

print("Model training completed")

model.save('MultiClassDefect.hdf5')
print("Saved model to disk with the name MultiClassDefect..")
print("Please execute EvaluatingMultiClassDefect file for the results")


import matplotlib.pyplot as plt

plt.plot(model_fit.history['categorical_accuracy'])
plt.plot(model_fit.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(model_fit.history['loss'])
plt.plot(model_fit.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('loss.png')


# %%
import numpy as np # linear algebra
import pandas as pd #
import glob
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import csv
import os
from keras.models import model_from_json
from keras.models import load_model

print("Loading data for training...")
img_size = (224, 224)
img_array_list = []
cls_list = []

train_dir = 'input/DataAugmentation/DataSplit/train/Class'
for i in range(6):
    x = str(i + 1)
    img_list1 = glob.glob(train_dir + x + '_def/*.png')
    for i in img_list1:
        img = load_img(i, color_mode='grayscale', target_size=(img_size))
        img_array = img_to_array(img) / 255
        img_array_list.append(img_array)
        cls_list.append(1)

    img_list1 = glob.glob(train_dir + x + '/*.png')
    for i in img_list1:
        img = load_img(i, color_mode='grayscale', target_size=(img_size))
        img_array = img_to_array(img) / 255
        img_array_list.append(img_array)
        cls_list.append(0)



X_train = np.array(img_array_list)
y_train = np.array(cls_list)

print("Loading data completed")

print("Creating Model...")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import tensorflow.keras.optimizers as opt

model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 1)))
model.add(MaxPooling2D(pool_size=(8, 8)))
model.add(Dropout(rate=0.5))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(rate=0.5))
model.add(Flatten())
model.add(Dense(units=1024, activation='relu'))

model.add(Dense(units=1, activation='sigmoid'))
Nadam = opt.Nadam(lr=8e-4, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
model.compile(optimizer=Nadam, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

from keras.callbacks import History
history = History()

print("Started training the model the model")
fit = model.fit(X_train, y_train, epochs=20, batch_size=32, callbacks=[history])
print("Model training completed")

model_json = model.to_json()
with open("IndDefectSplitModel.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("IndDefectSplitModel.h5")
model.save('IndDefectSplitModel.hdf5')
print("Saved model to disk with the name IndDefectSplitModel..")
print("Please execute EvaluatingCNNModel file for the results by changing the model name")





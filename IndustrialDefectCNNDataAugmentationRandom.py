import numpy as np
import pandas as pd
import glob
import csv
import os
import tensorflow.keras.optimizers as opt
import sklearn
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

print("Loading data for training...")
img_size = (224, 224)
img_array_list = []
cls_list = []

train_dir = 'input/DataAugmentation/Data/Class'

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

X = np.array(img_array_list)
y = np.array(cls_list)

print("Loading data completed")

print("Creating Model...")

y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 1)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(rate=0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(rate=0.25))

model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(rate=0.25))

model.add(Dense(units=2, activation='softmax'))

Nadam = opt.Nadam(lr=8e-4, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
model.compile(optimizer=Nadam, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

from keras import callbacks

filename='model_train_new.csv'
csv_log=callbacks.CSVLogger(filename, separator=',', append=False)

early_stopping=callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=0, mode='min')

filepath="Best-weights-my_model-{epoch:03d}-{loss:.4f}-{accuracy:.4f}.hdf5"

checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [csv_log,early_stopping,checkpoint]

from keras.callbacks import History
history = History()

print("Started training the model the model")
hist = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
print("Model training completed")

model_json = model.to_json()
with open("IndDefectRandomModel.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("IndDefectRandomModel.h5")
model.save('IndDefectRandomModel.hdf5')
print("Saved model to disk with the name IndDefectRandomModel..")

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['accuracy']
val_acc=hist.history['val_accuracy']
xc=range(5)

print("Plotting Epochs vs Accuracy graph")
plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
plt.style.use(['classic'])

print("Plotting Epochs vs Loss graph")
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['accuracy']
val_acc=hist.history['val_accuracy']
xc=range(5)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
plt.style.use(['classic'])

print("Please execute EvaluatingCNNModel file for the results by changing the model name")




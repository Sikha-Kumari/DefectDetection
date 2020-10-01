import os, cv2
from tensorflow.keras.preprocessing import image
import numpy as np
from matplotlib import pyplot as plt
import glob
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report,confusion_matrix
import itertools
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from keras.models import model_from_json
from keras.models import load_model

print("Loading Model...")

loaded_model=load_model('IndDefectRandomModel.hdf5')

loaded_model.summary()

img_array_test_list = []
cls_test_list = []
img_size = (224, 224)
test_dir = 'input/test'

img_size = (224, 224)
resultListDefect = []
resultListNonDefect = []
for i in os.listdir(test_dir):
    img = load_img(test_dir + '/' + i, color_mode='grayscale', target_size=(img_size))
    X = img_to_array(img) / 255
    X = np.expand_dims(X, axis=0)
    X_test = np.array(X)

    prediction = loaded_model.predict(X_test)[:, 0] > 0.5

    if (prediction):
        resultListDefect.append(prediction)
        plt.imshow(img)
        title_obj = plt.title('Defect')  # get the title property handler
        plt.getp(title_obj)  # print out the properties of title
        plt.getp(title_obj, 'text')  # print out the 'text' property for title
        plt.setp(title_obj, color='r')
    else:
        resultListNonDefect.append(prediction)
        plt.imshow(img)
        title_obj = plt.title('Non Defect')  # get the title property handler
        plt.getp(title_obj)  # print out the properties of title
        plt.getp(title_obj, 'text')  # print out the 'text' property for title
        plt.setp(title_obj, color='g')
    plt.show()

print(len(resultListDefect))
print(len(resultListNonDefect))

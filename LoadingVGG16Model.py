import os
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import glob
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report,confusion_matrix

from keras.models import model_from_json
from keras.models import load_model
json_file = open('IndDefectVGG19Model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("IndDefectVGG19Model.h5")
print("Loaded model from disk")

loaded_model=load_model('IndDefectVGG19Model.hdf5')

loaded_model.summary()

img_array_test_list = []
cls_test_list = []
img_size = (224, 224)
test_dir = 'input/DataAugmentation/DataSplit/test/Class'
for i in range(6):
    x = str(i + 1)
    img_list1 = glob.glob(test_dir + x + '/*.png')
    for i in img_list1:
        img = load_img(i, target_size=(img_size))
        img_array = img_to_array(img) / 255
        img_array_test_list.append(img_array)
        cls_test_list.append(0)

    img_list1 = glob.glob(test_dir + x + '_def/*.png')
    for i in img_list1:
        img = load_img(i, target_size=(img_size))
        img_array = img_to_array(img) / 255
        img_array_test_list.append(img_array)
        cls_test_list.append(1)

X_test = np.array(img_array_test_list)
y_test = np.array(cls_test_list)

y_pred = loaded_model.predict(X_test)[:, 0]

y_prediction = []
for pred in y_pred:
    if pred < 0.5:
        y_prediction.append(0)
    else:
        y_prediction.append(1)

target_names = ['Non Defect','Defect']

print(classification_report(y_test, y_prediction,target_names=target_names))
print(confusion_matrix(y_test, y_prediction))

test_dir_def = 'input/DataAugmentation/DataSplit/test/Class2_def'
img_size = (224, 224)
resultListDefect = []
resultListNonDefect = []
for i in os.listdir(test_dir_def):
    img = load_img(test_dir_def + '/' + i, target_size=(img_size))
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
    #plt.show()

print(len(resultListDefect))
print(len(resultListNonDefect))

from tensorflow.keras.preprocessing import image
from matplotlib import pyplot as plt
import os, cv2
import numpy as np
import glob
import csv
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import model_from_json
from keras.models import load_model

print("Loading Model....")
loaded_model = load_model('MultiClassDefect.hdf5')
loaded_model.summary()

print("Loading the test data....")
img_array_test_list = []
cls_test_list = []
img_size = (224, 224)
test_dir = 'input/DataAugmentation/DataSplit/test/Class'
img_array_test_list = []
cls_test_list = []
for i in range(6):
    x = str(i + 1)

    img_list1 = glob.glob(test_dir + x + '_def/*.png')
    for i in img_list1:
        img = load_img(i, target_size=(img_size))
        img_array = img_to_array(img) / 255
        img_array_test_list.append(img_array)
        cls_test_list.append(x)

X_test = np.array(img_array_test_list)
y_test = np.array(cls_test_list)

print("Loading test data completed")

print("Predicting the test results")
predict = loaded_model.predict(X_test)[:, 0]


y_prediction = []
y_pred = np.argmax(predict)

target_names = ['Class 1 Defect',' Class 2 Defect', 'Class 3 Defect', 'Class 4 Defect', 'Class 5 Defect', 'Class 6 Defect']

print(classification_report(y_test, y_prediction,target_names=target_names))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = (confusion_matrix(y_test, y_prediction))

np.set_printoptions(precision=2)

plt.figure()

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix')
plt.show()

print("Saving the test results...")
filenames =[]
with open('MultiCLassModelResult.csv', 'w', newline='') as writeFile:
    writer = csv.writer(writeFile)
    filenames.append('filename')
    writer.writerow(filenames)
    for i in range(6):
        x = str(i + 1)
        for filename in os.listdir(test_dir + x + '_def'):
            filenames=[]
            name = 'Class' + x + '_def_' + filename
            filenames.append(name)
            writer.writerow(filenames)
writeFile.close()
submit = pd.read_csv('MultiCLassModelResult.csv')
submit['defect'] = predict
submit.to_csv('MultiCLassModelResult.csv', index=False)
print("Results are saved in MultiCLassModelResult csv file")
print("Evaluating MultiCLass model execution is completed")

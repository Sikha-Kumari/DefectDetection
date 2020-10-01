import numpy as np
import glob
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

from keras.utils.np_utils import to_categorical

img_size = (224, 224)
img_array_list = []
cls_list = []

train_dir = 'input/DataAugmentation/DataSplit/train/Class'

for i in range(6):
    x = str(i + 1)
    img_list1 = glob.glob(train_dir + x + '/*.png')
    for i in img_list1:
        img = load_img(i, target_size=(img_size))
        img_array = img_to_array(img) / 255
        img_array_list.append(img_array)
        cls_list.append(0)

    img_list1 = glob.glob(train_dir + x + '_def/*.png')
    for i in img_list1:
        img = load_img(i, target_size=(img_size))
        img_array = img_to_array(img) / 255
        img_array_list.append(img_array)
        cls_list.append(1)

X = np.array(img_array_list)
y = np.array(cls_list)

y = to_categorical(y)
print("Data Loading completed")

IMAGE_SIZE = [224, 224]
vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
print("Loaded vgg model")
# don't train existing weights
for layer in vgg.layers:
    layer.trainable = False

out = Flatten()(vgg.output)
prediction = Dense(2, activation='softmax')(out)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

print("Started trainig the model")
r = model.fit(X, y, epochs=10, batch_size=64)




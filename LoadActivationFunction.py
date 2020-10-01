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

loaded_model=load_model('IndDefectSplitModel.hdf5')

loaded_model.summary()

#test_image = cv2.imread('input/DataAugmentation/DataSplit/test/Class2_def/136.png')
img_path = 'input/DataAugmentation/DataSplit/test/Class2_def/136.png'
img = image.load_img(img_path, color_mode='grayscale', target_size=(224, 224))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255
print(img_tensor.shape)



from keras import models

layer_outputs = [layer.output for layer in loaded_model.layers[:9]]
# Extracts the outputs of the top 12 layers
activation_model = models.Model(inputs=loaded_model.input,
                                outputs=layer_outputs)  # Creates a model that will return these outputs, given the model input

activations = activation_model.predict(img_tensor)
first_layer_activation = activations[0]
print(first_layer_activation.shape)
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')

layer_names = []
for layer in loaded_model.layers[:9]:
    layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot

images_per_row = 16
for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
    n_features = layer_activation.shape[-1]  # Number of features in the feature map
    size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols):  # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                            :, :,
                            col * images_per_row + row]
            channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size: (col + 1) * size,  # Displays the grid
            row * size: (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()

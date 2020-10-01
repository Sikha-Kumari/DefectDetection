# Execute this file for creating augmented images

import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 5)

import numpy as np
import os
import glob
import cv2

#Visualising the data
def visualize(original, augmented):
  fig = plt.figure()
  plt.subplot(1,2,1)
  plt.title('Original image')
  plt.imshow(original)

  plt.subplot(1,2,2)
  plt.title('Augmented image')
  plt.imshow(augmented)

path = 'input/data/train/Class1_def/1.png'
image = cv2.imread(path)
flipped = tf.image.flip_left_right(image)
visualize(image, flipped)
plt.show()

print("Generating augmented images for training...")
train_dir = 'input/data/train/Class'
for i in range(6):
    x = str(i + 1)
    img_list1 = glob.glob(train_dir + x + '_def/*.png')
    for img in img_list1:
        load_img = cv2.imread(img)
        flipped = tf.image.flip_left_right(load_img)
        image_path = train_dir + x + '_def/flipped'+ img[34:]
        cv2.imwrite(image_path, np.array(flipped))
        updown = tf.image.flip_up_down(load_img)
        image_path = train_dir + x + '_def/updown'+ img[34:]
        cv2.imwrite(image_path, np.array(updown))
        bright = tf.image.adjust_brightness(load_img, 0.4)
        image_path = train_dir + x + '_def/bright'+ img[34:]
        cv2.imwrite(image_path, np.array(bright))
        rotated = tf.image.rot90(load_img)
        image_path = train_dir + x + '_def/rotated'+ img[34:]
        cv2.imwrite(image_path, np.array(rotated))
        saturated = tf.image.adjust_saturation(load_img, 10)
        image_path = train_dir + x + '_def/saturated'+ img[34:]
        cv2.imwrite(image_path, np.array(saturated))
        grayscaled = tf.image.rgb_to_grayscale(load_img, 10)
        image_path = train_dir + x + '_def/grayscaled'+ img[34:]
        cv2.imwrite(image_path, np.array(grayscaled))
print("Data Augmentation completed")

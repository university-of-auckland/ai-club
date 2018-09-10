 # pass an image folder as parameter - python classify_birds_in_a_folder.py C:\Users\rkmu182\Desktop\Work\Python\ML\bird\test_birds\bird_album1\
 
 # -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import scipy
import numpy as np
import argparse

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from pathlib import Path


parser = argparse.ArgumentParser(description='Decide if an image is a picture of a bird')
parser.add_argument('album_folder', type=str, help='The image image file to check')
args = parser.parse_args()


# Same network definition as before
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='bird-classifier.tfl.ckpt')
model.load("bird-classifier.tfl.ckpt-8880")

import os
import shutil

is_a_bird_folder = args.album_folder + "\\" + "is_a_bird"
os.mkdir(is_a_bird_folder)

is_not_a_bird_folder = args.album_folder + "\\" + "is_not_a_bird"
os.mkdir(is_not_a_bird_folder)

  
#for filename in os.listdir(args.album_folder):
for filename in os.listdir(args.album_folder):	
  filename = args.album_folder + filename
  if filename.find(".") > 0:
	  print("Now processing filename " + filename)
	  # Load the image file
	  img = scipy.ndimage.imread(filename, mode="RGB")
	  # Scale it to 32x32
	  img = scipy.misc.imresize(img, (32, 32), interp="bicubic").astype(np.float32, casting='unsafe')
	  # Predict
	  prediction = model.predict([img])

	  # Check the result.
	  is_bird = np.argmax(prediction[0]) == 1
	  
	  if is_bird:
	   shutil.copy(filename, is_a_bird_folder) 
	   #img = Image.open(filename)
	   #draw = ImageDraw.Draw(img)
	   #draw.text((0, 0),"Is likely a bird",(255,255,255))
	   #img.save(filename)
	   
	  else:
	   shutil.copy(filename, is_not_a_bird_folder) 

print("Image classfication done!")

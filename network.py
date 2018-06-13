#----------------------------import required modules--------------------------#

import os
import shutil
import random
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator



#---------------------------Arranging the data set-----------------------------#
#split the data set into three categories: train, validation
#and test in the ratio 80%:10%:10%
base_directory = "all_years_140x140"
categories = os.listdir(base_directory)
for category in categories:
  train_path = "data/train/" + category
  validation_path = "data/validation/" + category
  test_path = "data/test/" + category
  if not os.path.exists(train_path):
    os.makedirs(train_path)
  if not os.path.exists(validation_path):  
    os.makedirs(validation_path)
  if not os.path.exists(test_path):  
    os.makedirs(test_path)
  if os.path.isdir(base_directory + "/" + category):
    images = os.listdir(base_directory + "/" + category)
    random.shuffle(images)
    train_data = images[:int((len(images)-1)*0.8)]
    validation_data = images[len(train_data):int((len(images)-1)*0.9)]
    test_data = images[int((len(images)-1)*0.9):]
    for image in train_data:
      shutil.move(base_directory + "/" + category + "/" + image, train_path)
    for image in validation_data:
      shutil.move(base_directory + "/" + category + "/" + image, validation_path)
    for image in test_data:
          shutil.move(base_directory + "/" + category + "/" + image, test_path)
shutil.rmtree(base_directory)


#---------------------------Construct the Model-------------------------------#

#check if input shape is keras or theano style
if K.image_data_format() == 'channels_first':
      input_shape = (3, 140, 140)
else:
      input_shape = (140, 140, 3)

network = Sequential()
network.add(Conv2D(32, (3,3), input_shape = input_shape))
network.add(Activation('relu'))


#----------------------------import required modules--------------------------#
print("importing the required modules")

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
print("")
base_directory = "all_years_140x140"
if not os.path.isdir(base_directory):
  print("all_years_140x140 not found!")
else:
  print("splitting the data into train set, validation set and test set.")
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


#---------------------------Loading the data------------------------------#
print("")
print("loading the data")

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

validate_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory('data/train', 
                                                    target_size = (140, 140), 
                                                    batch_size = 10,
                                                    class_mode = 'binary')

validation_generator = validate_datagen.flow_from_directory('data/validation',
                                                      target_size = (140, 140),
                                                      batch_size = 10,
                                                      class_mode = 'binary')

test_generator = test_datagen.flow_from_directory('data/test', 
                                                  target_size = (140, 140),
                                                  batch_size = 10,
                                                  class_mode = 'binary')


#---------------------------Construct the network-------------------------------#
print("")
print("constructing the network")
#check if input shape is keras or theano style
if K.image_data_format() == 'channels_first':
      input_shape = (3, 140, 140)
else:
      input_shape = (140, 140, 3)

network = Sequential()

network.add(Conv2D(32, (3,3), input_shape = input_shape))
network.add(Activation('relu'))
network.add(MaxPooling2D(pool_size = (2, 2)))

network.add(Conv2D(32, (3,3), input_shape = input_shape))
network.add(Activation('relu'))
network.add(MaxPooling2D(pool_size = (2, 2)))

network.add(Conv2D(64, (3,3), input_shape = input_shape))
network.add(Activation('relu'))
network.add(MaxPooling2D(pool_size = (2, 2)))

network.add(Flatten())

network.add(Dense(64))
network.add(Activation('relu'))
network.add(Dropout(0.5))

network.add(Dense(22))
network.add(Activation('softmax'))

network.compile(loss = 'categorical_crossentropy', 
                optimizer = 'rmsprop',
                metrics = ['accuracy'])

#---------------------------Training the network-------------------------------#
print("")
print("training network")
hist = network.fit_generator(train_generator, 
                            steps_per_epoch = int(34155/10),
                            epochs = 20,
                            validation_data = validation_generator,
                            validation_steps = int(4267/10))

#---------------------------Saving the network---------------------------------#
model.save_weights('classifier_weights.h5')
model.save('classifier.h5')

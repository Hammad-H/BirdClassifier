#----------------------------import required modules--------------------------#
print("importing the required modules")

import os
import shutil
import numpy as np
import keras
from keras import backend as K
from keras import optimizers
from keras.callbacks import History
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Reshape
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt



#---------------------------Declaring parameters-----------------------------------#


img_width = 140
img_height = 140
batch_size = 10
num_training_images = 33201
num_validation_images = 4166
learning_rate = 0.0001


"""
#---------------------------Loading the data---------------------------------------#
print("")
print("loading the data")
categories = os.listdir("data/train")
#ensure tha the hiddenfiles such as .DS_store are not considered categories.
while len(categories) != 13:
  del categories[0]


train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

validate_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory('data/train', 
                                                    target_size = (img_width, img_height), 
                                                    batch_size = batch_size,
                                                    class_mode = 'categorical',
                                                    classes = categories)

validation_generator = validate_datagen.flow_from_directory('data/validation',
                                                      target_size = (img_width, img_height),
                                                      batch_size = batch_size,
                                                      class_mode = 'categorical',
                                                      classes = categories)

test_generator = test_datagen.flow_from_directory('data/test', 
                                                  target_size = (img_width, img_height),
                                                  batch_size = batch_size,
                                                  class_mode = 'categorical',
                                                  classes = categories)




"""
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
network.add(Conv2D(32, (3,3)))
network.add(Activation('relu'))
network.add(MaxPooling2D(pool_size = (2, 2)))
network.add(Dropout(0.15))

network.add(Conv2D(32, (3,3)))
network.add(Activation('relu'))
network.add(Conv2D(32, (3,3)))
network.add(Activation('relu'))
network.add(MaxPooling2D(pool_size = (2, 2)))
network.add(Dropout(0.15))

network.add(Conv2D(32, (3,3)))
network.add(Activation('relu'))
network.add(Conv2D(32, (3,3)))
network.add(Activation('relu'))
network.add(MaxPooling2D(pool_size = (2, 2)))
network.add(Dropout(0.15))

network.add(Flatten())

network.add(Dense(1024))
network.add(Activation('relu'))
network.add(Dropout(0.5))


network.add(Dense(13))
network.add(Activation('softmax'))
optim = optimizers.rmsprop(lr = learning_rate, decay = 1e-6)

network.compile(loss = 'categorical_crossentropy', 
                optimizer = optim,
                metrics = ['accuracy'])

print(network.summary())
#---------------------------Training the network-------------------------------#
print("")
print("training the network")
steps_per_epoch = len(train_generator)
validation_steps = len(validation_generator)
model = network.fit_generator(train_generator, 
                            steps_per_epoch = steps_per_epoch,
                            epochs = 30,
                            validation_data = validation_generator,
                            validation_steps = validation_steps)

#---------------------------Saving the network---------------------------------#

network.save_weights('classifier(final)_weights.h5')
network.save('classifier(final).h5')

#---------------------------Testing the network-------------------------------#
print("testing the network")

results = network.evaluate_generator(test_generator)
print("The test accuracy is:", results[1])
print("The test loss is:", results[0])
#--------------------------Plotting Accuracy and Loss---------------------------#

fig = plt.figure(figsize = (9, 3.5))
fig.suptitle('Bird Classifier training performace on all_years_140x140 data')

ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(model.history['loss'], label = 'Training loss')
ax1.plot(model.history['val_loss'], label = 'Validation loss')
ax1.set_title('Training/Validation Loss')
ax1.set(ylabel = 'Loss')
ax1.set(xlabel = 'Epoch')
ax1.legend()

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(model.history['acc'], label = 'Training accuracy')
ax2.plot(model.history['val_acc'], label = 'Validation accuracy')
ax2.set_title('Training/Validation Accuracy')
ax2.set(ylabel = 'Accuracy')
ax2.set(xlabel = 'Epoch')
ax2.legend()

plt.show
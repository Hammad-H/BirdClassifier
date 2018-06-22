from keras.models import load_model
import os
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

categories = os.listdir("data/train")
#ensure tha the hiddenfiles such as .DS_store are not considered categories.
while len(categories) != 10:
  del categories[0]

network = load_model('classifier(deep).h5')
test_datagen = ImageDataGenerator(rescale = 1./255)
test_generator = test_datagen.flow_from_directory('data/test', 
                                                  target_size = (140, 140),
                                                  batch_size = 10,
                                                  class_mode = 'categorical',
                                                  classes = categories)

results = network.evaluate_generator(test_generator)
print("The test accuracy is:", results[1])
print("The test loss is:", results[0])
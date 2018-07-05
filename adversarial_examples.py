import keras
from keras.models import load_model
from keras.preprocessing import image 
from keras import optimizers
import foolbox
from foolbox.models import KerasModel
from foolbox.attacks import ProjectedGradientDescentAttack
from foolbox.criteria import TargetClassProbability
from foolbox.distances import Linfinity
from foolbox.adversarial import Adversarial
import numpy as np 
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import imageio
import os
import PIL.Image
import shutil
import random
from random import randint

def generate_adversarial(file_name, original_path, original_class, categories):
  original_image = mpimg.imread(original_path)
  target_class = randint(0, 12)
  while target_class == original_class:
    target_class = randint(0, 12)
  criteria = TargetClassProbability(target_class, 0.95)
  adversarial = Adversarial(adversarial_model,
                            criterion = criteria,
                            original_image = original_image,
                            original_class = original_class,
                            distance = Linfinity)
  adversarial_image = attack(adversarial,
                             epsilon = 0.0003,
                             random_start = True,
                             iterations = 200,
                             stepsize = 0.0001,
                             return_early = False)
  if adversarial_image is None:
    return
  adverse = np.round(adversarial_image)
  original_category = categories[original_class]
  target_category = categories[target_class]
  original_image_name = file_name[:-4]
  save_path = "adversarial_examples/" + original_category + "/"
  saved_image_name = original_image_name + "_" + target_category + ".png"
  matplotlib.image.imsave(save_path+saved_image_name, adverse/255, format = 'png')


#----------------creating the adversarial Model------------------------#
keras.backend.set_learning_phase(0)
network = load_model("classifier(new_data).h5")
adversarial_model = KerasModel(network, 
                              bounds = (0, 255), 
                              preprocessing = (0, 1))
attack = ProjectedGradientDescentAttack()


#----------------------------Creating the directories------------------#
base_directory = "all_years_140x140"
categories = os.listdir(base_directory)
while len(categories) != 13:
  del categories[0]
for category in categories:
  adversary_path = "adversarial_examples/" + category
  if not os.path.exists(adversary_path):
    os.makedirs(adversary_path)
  if os.path.isdir(base_directory + "/" + category):
      images = os.listdir(base_directory + "/" + category)
      random.shuffle(images)
      adversarial_originals = images[:int(len(images)*0.4)]
      for image in adversarial_originals:
        image_path = base_directory + "/" + category + "/" + image
        category_index = categories.index(category)
        generate_adversarial(image ,image_path, category_index, categories)
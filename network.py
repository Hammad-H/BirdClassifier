import os
import shutil
import random

#-------------Arranging the data set--------------#
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
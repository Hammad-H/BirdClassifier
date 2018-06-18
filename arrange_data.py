#This program takes the bird data set arranges it into three categories
#train validation and test.

import os
import shutil
import random
import tarfile


#---------------------------Extracting the tgz file----------------------------#

tar = tarfile.open("all_years_through_2017_140x140.tgz")
tar.extractall()
tar.close()

#---------------------------Arranging the data set-----------------------------#
#split the data set into three categories: train, validation
#and test in the ratio specified by the user.

print("")
base_directory = "all_years_140x140"
if not os.path.isdir(base_directory):
  print("all_years_140x140 not found!")
else:
  print("splitting the data into train set, validation set and test set.")
  train_ratio = int(input("Enter the percentage of data set to be used for " + 
                              "training: "))
  validation_ratio = int(input("Enter the percentage of data set to be used for " + 
                              "validation: "))
  test_ratio = 100 - (train_ratio + validation_ratio)
  print("Percentage of data used for test: " + str(test_ratio))
  train_ratio = train_ratio/100
  validation_ratio = validation_ratio/100
  test_ratio = test_ratio/100
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
      train_data = images[:int((len(images)-1)*train_ratio)]
      validation_data = images[len(train_data):int((len(images)-1)*(train_ratio + validation_ratio))]
      test_data = images[int((len(images)-1)*(train_ratio + validation_ratio)):]
      for image in train_data:
        shutil.move(base_directory + "/" + category + "/" + image, train_path)
      for image in validation_data:
        shutil.move(base_directory + "/" + category + "/" + image, validation_path)
      for image in test_data:
            shutil.move(base_directory + "/" + category + "/" + image, test_path)
  shutil.rmtree(base_directory)


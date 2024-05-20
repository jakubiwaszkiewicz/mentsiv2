import torch
from torch import nn
import os

import random
from PIL import Image # type: ignore
import glob
from pathlib import Path

import numpy as np # type: ignore

import matplotlib.pyplot as plt

from sklearn.base import clone

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from tqdm import tqdm
# from time import sleep


# for iteration_id, iteration in tqdm(enumerate(range(1000)), total=1000):
#   sleep(0.2)


# exit();


train_dir = "dogs_vs_cats/train"
test_dir = "dogs_vs_cats/test"



# Set seed
random.seed(1410)

# 1. Get all image paths (* means "any combination")

# importowanie do pliku danych wejsciowych z zew. folderu
train_image_path_list= glob.glob(f"{train_dir}/*/*.jpg")
test_image_path_list= glob.glob(f"{test_dir}/*/*.jpg")

# mieszanie ścieżek do danych testowych i treningowych
random.shuffle(test_image_path_list)
random.shuffle(train_image_path_list)


# Limiting datasets to 2000 and 500 items i alokuje miejsce
limited_train_image_path = np.zeros(2000, dtype=('U', 255))
limited_test_image_path = np.zeros(500, dtype=('U', 255))

# petla do zlimitowania ilosci zdjec w tablicach np
for idx, train_image_path in enumerate(train_image_path_list):
  limited_train_image_path[idx] = train_image_path
  if idx == 1999:
    break

for idx, test_image_path in enumerate(test_image_path_list):
  limited_test_image_path[idx] = test_image_path
  if idx == 499:
    break


# Initializing datasets

# tensor 3 wymiarowy, kostka zawierajaca zdjecia
train_X = np.zeros((2000,100,100), dtype=int)
train_y = np.zeros(2000, dtype=int)

test_X = np.zeros((500,100,100), dtype=int)
test_y = np.zeros(500, dtype=int)

# zainicjalizowanie spłaszczonej wersji danych
train_X_flatten = np.zeros((2000,10000), dtype=int)
test_X_flatten = np.zeros((500,10000), dtype=int)


#preparing train data
for idx, train_path in tqdm(enumerate(limited_train_image_path), total=2000):
  # if idx == randint:
  #   print(train_path)
  img_label = Path(train_path).parent.stem
  if img_label == "dogs":
    img_label = 0
  if img_label == "cats":
    img_label = 1

  # opening img and transforming img to grayscale
  ximg = Image.open(train_path).convert('L')

  # dostosowanie zdjecia do formatu pixeli 100x100
  ximg = ximg.resize((100, 100))


  ximg = np.asarray(ximg)
  train_X[idx] = ximg
  ximg = ximg.flatten()
  train_X_flatten[idx] = ximg
  train_y[idx] = img_label

for idx, test_path in tqdm(enumerate(limited_test_image_path), total=500):
  # if idx == randint:
  #   print(test_path)
  img_label = Path(test_path).parent.stem
  if img_label == "dogs":
    img_label = 0
  if img_label == "cats":
    img_label = 1

  # Transforming img to grayscale
  ximg = Image.open(test_path).convert('L')
  ximg = ximg.resize((100, 100))

  ximg = np.asarray(ximg)
  test_X[idx] = ximg
  ximg = ximg.flatten()
  test_X_flatten[idx] = ximg
  test_y[idx] = img_label

  #make arr flat to bring it to clf



clfs = {
  'GNB': GaussianNB(),
  'KNN': KNeighborsClassifier(3),
  'MLP': MLPClassifier(hidden_layer_sizes=((30,30,30))),
}

metric = accuracy_score

score = np.zeros(len(clfs))

for idx, clf in enumerate(clfs):
  clfs[clf].fit(train_X_flatten, train_y)
  y_pred = clfs[clf].predict(test_X_flatten)
  score[idx] = metric(test_y, y_pred)
  print(f"{clf} has score: {score[idx]}")
  print("Another classifier...")

print(score)
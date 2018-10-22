import tensorflow as tf
#KERAS
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras.models import model_from_json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import theano
from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

X_test = np.load('X_test.npy')
Y_test = np.load('Y_test.npy')

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X_test, Y_test, verbose=0) # accuracy check
print('Test score:', score[0])
print('Test accuracy:', score[1]*100, '%')
print('There are 2000 images shuffled for testing, how many do you want to test?')
num = input('Number of images to test: ')

for count in range(int(num)):
	value = input('Input random number between 0 and 2000: ')
	value = int(value)
	print('Predicted value: ')
	print(loaded_model.predict_classes(X_test[value:value+1]))
	print('Expected value: ')
	print(Y_test[value])

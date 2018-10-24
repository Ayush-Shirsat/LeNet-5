# Used to test data

import tensorflow as tf
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Disable warning message of tensorflow
import theano
from PIL import Image
from numpy import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

# Load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# Load X_test and Y_test
X_test = np.load('X_test.npy')
Y_test = np.load('Y_test.npy')

# List of classes used for classification
target_names = ['Dollar', 'Pound', 'Euro', 'Indian Rupee', 'Yen']

# Evaluate loaded model on test data
opt = SGD(lr = 0.01)
loaded_model.compile(loss='categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
score = loaded_model.evaluate(X_test, Y_test, verbose=0) # accuracy check

print('Test accuracy:', score[1], '%') # Prints test accuracy
print('\nThere are 2000 images shuffled for testing, how many do you want to test?')

# Users have the option to select number of images to test
num = input('Number of images to test: ')
print('\nConfidences are as follows:')

# Use the following list as reference while checking confidences
print('[Dollar	Pound	Euro	Indian Rupee	Yen]')
for count in range(int(num)):

	# User picks a random image from test data of size 2000
	value = input('\nInput random number between 0 and 2000: ')
	value = int(value)

	# If statement is added to handle error if user inputs incorrect value
	if 0 <= value < 2000: 
		pred = loaded_model.predict_classes(X_test[value:value+1])
		print('\n\nPredicted symbol: ', target_names[pred[0]]) # Prints models Predicted value 
		print('Expected symbol: ', target_names[np.argmax(Y_test[value])]) # Prints Expected value
		print('Confidences: ', loaded_model.predict(X_test[value:value+1])) # Prints confidence of each symbol
	else:
		print('Input number out of test range')

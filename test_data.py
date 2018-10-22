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
target_names = ['Dollar', 'Pound', 'Euro', 'Indian Rupee', 'Yen']

# evaluate loaded model on test data
opt = SGD(lr = 0.01)
loaded_model.compile(loss='categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
score = loaded_model.evaluate(X_test, Y_test, verbose=0) # accuracy check
print('\nTest score:', score[0])
print('Test accuracy:', score[1]*100, '%')
print('\nThere are 2000 images shuffled for testing, how many do you want to test?')
num = input('Number of images to test: ')
print('\nConfidences are as follows:')
print('[Dollar	Pound	Euro	Indian Rupee	Yen]')
for count in range(int(num)):
	value = input('\nInput random number between 0 and 2000: ')
	value = int(value)
	if 0 <= value < 2000: 
		pred = loaded_model.predict_classes(X_test[value:value+1])
		print('\n\nPredicted value: ', target_names[pred[0]])
		print('Expected value: ', target_names[np.argmax(Y_test[value])])
		print('Confidences: ', loaded_model.predict(X_test[value:value+1]))
	else:
		print('Input number out of test range')

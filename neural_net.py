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

import warnings
warnings.filterwarnings("ignore")

# input image dimensions
img_rows, img_cols = 28, 28

# number of channels
img_channels = 1

path1 = '/home/ece-student/EC_601/Mini_Proj_2/mini_proj_2/dataset/'    #path of folder of images      

listing = os.listdir(path1) 
num_samples = size(listing)
print(num_samples)

for file in listing:
    im = Image.open(path1 + file)   
    img = im.resize((img_rows,img_cols))
    gray = img.convert('L')          
    gray.save(path1 +  file, "JPEG")

imlist = os.listdir(path1)
imlist = sorted(imlist)

im1 = array(Image.open(path1 + imlist[0])) # open one image to get size
m,n = im1.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images

# create matrix to store all flattened images
immatrix = array([array(Image.open(path1 + im2)).flatten()
              for im2 in imlist],'f')
                
label = np.ones((num_samples,),dtype = int)
label[0:2000] = 0
label[2000:4000] = 1
label[4000:6000] = 2
label[6000:8000] = 3
label[8000:10000] = 4

data,Label = shuffle(immatrix,label, random_state = 4)

#batch_size to train
batch_size = 256
# number of output classes
nb_classes = 5
# number of epochs to train
nb_epoch = 100


# number of convolutional filters to use
nb_filters_1 = 20
nb_filters_2 = 50
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 5

(x, y) = (data, Label)


#split x and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 4)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Implementing a LeNet model
model = Sequential()

model.add(Convolution2D(nb_filters_1, kernel_size = (nb_conv, nb_conv), activation='relu', input_shape=(img_rows, img_cols, 1)))
model.add(MaxPooling2D(pool_size = (nb_pool, nb_pool), strides = (2, 2)))

model.add(Convolution2D(nb_filters_2, kernel_size = (nb_conv, nb_conv), activation='relu'))
model.add(MaxPooling2D(pool_size = (nb_pool, nb_pool), strides = (2, 2)))

model.add(Flatten())
model.add(Dense(500, activation='relu'))

model.add(Dense(nb_classes, activation='softmax'))

opt = SGD(lr = 0.01)
model.compile(loss='categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
         
      
hist = model.fit(X_train, Y_train, batch_size = batch_size, epochs = nb_epoch, verbose = 1, validation_split = 0.25, shuffle = True)

# visualizing losses and accuracy

train_loss = hist.history['loss']
val_loss = hist.history['val_loss'] 
train_acc = (hist.history['acc'])
val_acc = (hist.history['val_acc'])
# xc = range(nb_epoch)

score = model.evaluate(X_test, Y_test, verbose=0) # accuracy check
print('\nTest score:', score[0])
print('Test accuracy:', score[1])

# Confusion Matrix
from sklearn.metrics import classification_report,confusion_matrix

Y_pred = model.predict(X_test)
#print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
#print(y_pred)  
y_pred = model.predict_classes(X_test)
#print(y_pred) 

p = model.predict_proba(X_test) # to predict probability
print('\nConfusion Matrix')
print(confusion_matrix(np.argmax(Y_test,axis=1), y_pred))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 
np.save('X_test', X_test)
np.save('Y_test', Y_test)

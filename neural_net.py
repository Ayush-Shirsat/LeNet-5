import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,adam
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image
# SKLEARN
from sklearn.utils import shuffle
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split

import warnings
warnings.filterwarnings("ignore")

# input image dimensions
img_rows, img_cols = 28, 28

# number of channels
img_channels = 1

path1 = '/home/ece-student/EC_601/Mini_Proj_2/mini_proj_2/dataset/'    #path of folder of images    
path2 = '/home/ece-student/EC_601/Mini_Proj_2/mini_proj_2/dataset_gray/'  #path of folder to save images    

listing = os.listdir(path1) 
num_samples = size(listing)
print(num_samples)

for file in listing:
    im = Image.open(path1 + file)   
    img = im.resize((img_rows,img_cols))
    gray = img.convert('L')          
    gray.save(path2 +  file, "JPEG")

imlist = os.listdir(path2)
imlist = sorted(imlist)

im1 = array(Image.open(path2 + imlist[0])) # open one image to get size
m,n = im1.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images

# create matrix to store all flattened images
immatrix = array([array(Image.open(path2 + im2)).flatten()
              for im2 in imlist],'f')
                
label=np.ones((num_samples,),dtype = int)
label[0:2000] = 0
label[2000:4000] = 1
label[4000:6000] = 2
label[6000:8000] = 3
label[8000:10000] = 4

data,Label = shuffle(immatrix,label, random_state = 4)
#data = immatrix
#Label = label
train_data = (data,label)

#print(train_data[0].shape)
#print(train_data[1].shape)

#batch_size to train
batch_size = 50
# number of output classes
nb_classes = 5
# number of epochs to train
nb_epoch = 5


# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 5

#(x, y) = (train_data[0],train_data[1])
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

model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size=(nb_conv, nb_conv), activation='relu', input_shape=(img_rows, img_cols, 1)))
model.add(Convolution2D(nb_filters*2, (nb_conv, nb_conv), activation='relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))
'''
model.add(Convolution2D(nb_filters, (nb_conv, nb_conv), padding='same', input_shape=(img_rows, img_cols, 1)))

convout1 = Activation('sigmoid')
model.add(convout1)
model.add(Convolution2D(nb_filters, (nb_conv, nb_conv)))
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes)) 
model.add(Activation('softmax')) '''
model.compile(loss='categorical_crossentropy', optimizer = 'Adadelta', metrics = ['accuracy'])

#hist = model.fit(X_train, Y_train, batch_size = batch_size, epochs = nb_epoch, verbose=1, validation_data=(X_test, Y_test))            
with tf.device('/GPU:0'):       
  hist = model.fit(X_train, Y_train, batch_size = batch_size, epochs = nb_epoch, verbose = 1, validation_split = 0.25, shuffle = False)

# visualizing losses and accuracy

  train_loss=hist.history['loss']
  val_loss=hist.history['val_loss'] 
  train_acc=hist.history['acc']
  val_acc=hist.history['val_acc']
  xc = range(nb_epoch)

  score = model.evaluate(X_test, Y_test, verbose=0) # accuracy check
  print('Test score:', score[0])
  print('Test accuracy:', score[1])
  print(model.predict_classes(X_test[45:50]))
  print(Y_test[45:50])

# Confusion Matrix

from sklearn.metrics import classification_report,confusion_matrix

Y_pred = model.predict(X_test)
#print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
#print(y_pred)  
y_pred = model.predict_classes(X_test)
#print(y_pred) 

p = model.predict_proba(X_test) # to predict probability

target_names = ['class 0(Dollar)', 'class 1(Pound)', 'class 2(Euro)', 'class 3(Indian Rupee)', 'class 4(Yen)']
print(confusion_matrix(np.argmax(Y_test,axis=1), y_pred))

# saving weights

'''fname = "weights-Test-CNN.txt"
model.save_weights(fname,overwrite=True)

# Loading weights

fname = "weights-Test-CNN.txt"
model.load_weights(fname)'''

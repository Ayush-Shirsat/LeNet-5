# Mini Project-2: Classification of handwritten currency symbols

## Built using
*Python 3.6.5*

*Tensorflow 1.11.0*

*Keras 2.2.4*

*Numpy 1.13.0*

*Matplotlib 2.2.2*

*Theano 1.0.3*

*Scikit-learn 0.20.0*

## Dataset

There are 5 classes (or handwritten symbols) to distinguish namely Dollar, Pound, Euro, Indian Rupee and Yen.

Images were generated using MATLAB. 400 symbols (for each class) were written on a sheet of paper and extracted using image processing techniques. 2000 (400 x 5) symbols were detected and put in a bounding box. Each of the bounding box is an image containing a symbol. Each of this image was rotated by 0,5,-5,10,-10 degrees to increase the size of dataset. All symbols were mapped on a gray scale image and resized to size 28x28. 

Total images generated: 10000 (2000 for each class)

Link to dataset: https://drive.google.com/open?id=1jbGHiqryq0917xHMBEDg51pBkrqeAaCX (use @bu.edu to access)

## Neural Net

A LeNet-5 model was implemented.

Total images: 10000

Training set: 6000

Testing set: 2000

Validation set: 2000

The sequential model is as follows:

Conv → Pooling → Conv → Pooling → Fully connected → Fully connected → Output

## Results

Training loss: 0.0157

Training accuracy: 99.72%

Validation loss: 0.0129

Validation accuracy: 99.45%

Total epochs:100

Batch size: 256

Test accuracy: 99.1%


## Confusion Matrix:

**[387  2   0   0   0]**

**[0   426  0   0   0]**
 
**[0   11  380  0   1]**
 
**[2   0   0  401   0]**
 
**[1   0   1   0  388]**

## Errors/ Bugs

While testing the program you could face warnings which are related to library versions. Make sure libraries installed matched with the ones mentioned before. You could also ignore these warnings, code should work fine regardless.

## How to test the program

There is no need to run the neural net model. The weights and model are already saved. Make sure the following files are in your directory:

* neural_net.py

* test_data.py

* X_test.npy

* Y_test.npy

* model.h5

* model.json

**Optional**

The model is already pre-trained so you do not have to train it again. However, if you wish to train your model download the dataset from google drive link mentioned above. Make sure your file path on ``` line: 29 ``` is correct

Run the following command in your terminal:
```
python neural_net.py
```

**Test the program**

Run the following command in your terminal:
```
python test_data.py
```

Follow the on screen instruction and check the results.

The program give the Predicted symbol, Expected symbol and Confidence. 

List of classes: [Dollar, Pound, Euro, Indian Rupee, Yen]

Compare these classes with the Confidence value to check if Predicted values are correct.

## References

* http://notesofaprogrammer.blogspot.com/2018/03/pymc3-reports-warning-theanotensorblas.html

* https://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/

* https://www.tensorflow.org/tutorials/keras/basic_classification

* https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.save.html

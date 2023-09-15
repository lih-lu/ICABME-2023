"""Synthetic PET images
#Author: Selma Boudissa
# Data parallel: Ezhilmathi Krishnasamy
#Date: 29/10/2021
OB#Purpose: - Load mri and fmiso numpy array for train and test
- Split the dataset into train and validation set
- Build the neural network
- Training of the network
- Plot the learning curve
- Evaluate and predict on test set to check the performance of the model
"""

# import needed libraries
import tensorflow as tf # Google's framework to build AI and ML models
import pandas as pd # used for dataframe manipulation
import matplotlib.pyplot as plt # used for data visualization and plotting
import numpy as np # used for numerical analysis
import Models # script where all the models are saved (2D/3D Unet and OCM)
import keras.backend as K
from Models import create_unet_model2D, create_unet_model3D
from keras import regularizers
from keras import backend as K
from keras.layers.core import Dense
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from keras.applications import imagenet_utils
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D,Conv3D, Flatten, MaxPooling2D,MaxPooling3D, Dropout, BatchNormalization, InputLayer
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from keras.layers import *
from keras.models import *
from tqdm.notebook import trange, tqdm
from matplotlib import pyplot, cm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv2D,Conv3D,UpSampling2D,UpSampling3D,MaxPooling2D,MaxPooling3D,Concatenate,Dropout,Input)
from keras.models import Model
import keras
import math
import horovod.tensorflow.keras as hvd
import keras
import keras.utils


# Initialize Horovod
hvd.init()


# Pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    
print('load data.....')
X= np.load('MRI-reconstruction-SyntheticPET/new-data/X.npy') # precise the correct path to load your data
y= np.load('MRI-reconstruction-SyntheticPET/new-data/y.npy') # precise the correct path to load your data


print('load test data...')
X_test= np.load('MRI-reconstruction-SyntheticPET/data/3D/test/X_test.npy')
y_test= np.load('MRI-reconstruction-SyntheticPET/data/3D/test/y_test.npy')


# split data for train and validation
X_train,X_valid, y_train, y_valid= train_test_split (X, y, test_size= 0.1 , random_state = 42)


# info for callbacks 
date = '2910'
name = '3DUnet'
sample = 'brains'
batchsize = 4 # 4 original model
#epochs = 50 # modifiy number of epochs here
epochs = int(math.ceil(50.0 / hvd.size()))
layers = 3


# Build network model
# call model function for 3D Unet
Unet_3D_model  = create_unet_model3D(input_image_size=(240,240,128,4), n_labels=1, layers=3,
                            mode='regression', output_activation='relu')
model = Unet_3D_model 


print('define callbacks...')
callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)] 


# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(keras.callbacks.ModelCheckpoint('./{}.{}.{}.{}.{}.{}'.format(date, name, batchsize, epochs,sample,layers) + '.weights.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=0)), 

    
print('train model......')
synpet = model.fit(X_train,y_train,validation_data=(X_valid, y_valid),
                   epochs = epochs, 
                   batch_size=batchsize, 
                   verbose = 1 if hvd.rank() == 0 else 0, 
                   shuffle= True, 
                   callbacks = callbacks) 


# evaluate model                                                 
test_loss = model.evaluate(X_test,y_test, verbose=1)
print("test_loss:{}".format(test_loss))


# plot learning curve 
fig = plt.figure() # using 3 layers
plt.plot(synpet.history['loss'])
plt.plot(synpet.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
fig.savefig('model_loss_{}.{}.{}.{}.{}.jpg'.format(date, name, batchsize, epochs,sample))


print('prediction...')
synthetic_PET = model.predict(X_test)


# Running project completed!
print('Done!')

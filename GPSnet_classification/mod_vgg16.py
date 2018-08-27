from __future__ import division, print_function

import warnings
import numpy as np
import os
import csv

from keras import backend as K
from keras import layers
from keras.layers.core import Activation, Dense, Flatten
from keras.models import Model
from keras.layers import Input, Conv2D, ZeroPadding2D, BatchNormalization, MaxPooling2D
from keras.regularizers import l2
from keras.layers.core import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.utils import plot_model
from scipy.misc import imread, imresize
import tensorflow as tf

from vgg16_places_365 import VGG16_Places365

#Specifying GPU usage.
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]="3"


classes = 121

def create_my_model():
    #Get back the convolutional part of a VGG network trained on ImageNet
    img_input = Input(shape=(224,224,3), name = 'image_input')
    my_model = VGG16_Places365(weights= 'places', input_tensor = img_input, include_top = False, classes = classes)
    
    #fix vgg16 convnet weight as untrainable (1. initializer 2. feature extracter)

    for layer in my_model.layers:
        layer.trainable = False
    
    #Add the fully-connected layers 
    output_base_model = my_model.layers[-1].output
    x = Flatten(name='flatten')(output_base_model)
    
    x = BatchNormalization(name = 'bn_flatten')(x)
    #x = Dropout(0.4, name='drop_flatten')(x)
    x = Dense(128, activation='relu', kernel_initializer= 'he_normal', name='fc1')(x)
    x = BatchNormalization(momentum = 0.9, name='bn1')(x)
    #x = Dropout(0.4, name='drop_fc1')(x)
    x = Dense(128, activation='relu', kernel_initializer= 'he_normal', name='fc2')(x)
    x = BatchNormalization(momentum = 0.9, name='bn2')(x)
    #x = Dropout(0.4, name='drop_fc2')(x)

    predicted_img_class = Dense(classes, activation='softmax', name='predictions')(x)
    
    #Create your own model
    my_model = Model(input = img_input, output = predicted_img_class)
    
    #In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
    my_model.summary()
    plot_model(my_model, to_file='model_train.png', show_shapes = True)
    return my_model

def create_my_model_without_dropout():
    #Get back the convolutional part of a VGG network trained on ImageNet
    img_input = Input(shape=(224,224,3), name = 'image_input')
    my_model = VGG16_Places365(weights= 'places', input_tensor = img_input, include_top = False, classes = classes)


    
    #Add the fully-connected layers 
    output_base_model = my_model.layers[-1].output
    x = Flatten(name='flatten')(output_base_model)
    
    x = BatchNormalization(name = 'bn_flatten')(x)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    x = BatchNormalization(name='bn2')(x)
    
    predicted_img_class = Dense(classes, activation='softmax', name='predictions')(x)
    
    #Create your own model
    my_model = Model(input = img_input, output = predicted_img_class)
    
    for layer in my_model.layers:
        layer.trainable = False

    #In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
    my_model.summary()
    plot_model(my_model, to_file='model_test.png', show_shapes = True)
    return my_model
    
    
if __name__ == "__main__":
    print("Please run either test.py or train.py to evaluate or fine-tune the network!")

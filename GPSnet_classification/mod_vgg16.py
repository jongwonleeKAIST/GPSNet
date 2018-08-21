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


def euc_loss(true, pred):
    lx = K.sqrt(K.sum(K.square(true[:,:] - pred[:,:]), axis=1, keepdims=True))
    #print "true: ", true
    #print "pred: ", pred
    #   print "euc loss: ", lx
    return (lx)


# returns mean error of each data sequence used by traing and test.
def threshold(unsuffled_csv_data_path):
    with open(unsuffled_csv_data_path) as f:
        error_list = []

        csvreader = csv.reader(f, delimiter = ' ')
        next(csvreader)
        next(csvreader)
        next(csvreader)
        firstline = next(csvreader)

        prev_utm = np.asarray([int(firstline[3]), int(firstline[4])])

        next(csvreader)
        for line in csvreader:
            current_utm = np.asarray([int(line[3]), int(line[4])])
            error = np.linalg.norm(current_utm - prev_utm)
            error_list.append(error)
            prev_utm = current_utm

        return reduce(lambda x, y: x + y, error_list) / len(error_list)



#Specifying GPU usage.
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]="3"


def create_my_model():
    #Get back the convolutional part of a VGG network trained on ImageNet
    img_input = Input(shape=(224,224,3), name = 'image_input')
    my_model = VGG16_Places365(weights= 'places', input_tensor = img_input, include_top = False)
    
    #fix vgg16 convnet weight as untrainable (1. initializer 2. feature extracter)

    for layer in my_model.layers:
        layer.trainable = False
    
    #Use the generated model
    output_base_model = my_model.layers[-6].output
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name="block4_pool", padding='valid')(output_base_model)
    x = Flatten(name='flatten')(x)
    
    #Add the fully-connected layers 
    #x = Flatten(name='flatten')(output_base_model)
    x = BatchNormalization(name = 'bn_flatten')(x)
    #x = Dropout(0.4, name='drop_flatten')(x)
    x = Dense(256, activation='relu', kernel_initializer= 'he_normal', name='fc1')(x)
    x = BatchNormalization(momentum = 0.9, name='bn1')(x)
    #x = Dropout(0.4, name='drop_fc1')(x)
    x = Dense(256, activation='relu', kernel_initializer= 'he_normal', name='fc2')(x)
    x = BatchNormalization(momentum = 0.9, name='bn2')(x)
    #x = Dropout(0.4, name='drop_fc2')(x)
    x = Dense(256, activation='relu', kernel_initializer= 'he_normal', name='fc3')(x)
    x = BatchNormalization(momentum = 0.9, name='bn3')(x)
    #x = Dropout(0.4, name='drop_fc3')(x)   
    
    fc_pose_utmx_utmy = Dense(2, name = 'fc_pose_utmx_utmy')(x)
    
    #Create your own model
    my_model = Model(input = img_input, output = fc_pose_utmx_utmy)
    
    #In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
    my_model.summary()
    plot_model(my_model, to_file='model_train.png', show_shapes = True)
    return my_model

def create_my_model_without_dropout():
    #Get back the convolutional part of a VGG network trained on ImageNet
    img_input = Input(shape=(224,224,3), name = 'image_input')
    my_model = VGG16_Places365(weights= 'places', input_tensor = img_input, include_top = False)

    #Use the generated model 
    output_base_model = my_model.layers[-6].output
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name="block4_pool", padding='valid')(output_base_model)
    x = Flatten(name='flatten')(x)
    
    #Add the fully-connected layers 
    #x = Flatten(name='flatten')(output_base_model)
    x = BatchNormalization(name = 'bn_flatten')(x)
    x = Dense(256, activation='relu', name='fc1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = Dense(256, activation='relu', name='fc2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = Dense(256, activation='relu', name='fc3')(x)
    x = BatchNormalization(name='bn3')(x)
    
    fc_pose_utmx_utmy = Dense(2, name = 'fc_pose_utmx_utmy')(x)
    
    #Create your own model
    my_model = Model(input = img_input, output = fc_pose_utmx_utmy)
    
    for layer in my_model.layers:
        layer.trainable = False

    #In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
    my_model.summary()
    plot_model(my_model, to_file='model_test.png', show_shapes = True)
    return my_model
    
    
if __name__ == "__main__":
    print("Please run either test.py or train.py to evaluate or fine-tune the network!")

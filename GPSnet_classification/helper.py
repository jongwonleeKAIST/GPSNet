from tqdm import tqdm
import numpy as np
import os.path
import sys
import random
import math
import cv2


def init(seq_name):
    global directory, dataset_train, dataset_test
    directory = '../img_files/'
    dataset_train = '{}_train.csv'.format(seq_name)
    dataset_test = '{}_test.csv'.format(seq_name)


class datasource(object):
    def __init__(self, images, poses, filenames, classes):
        self.images = images
        self.poses = poses
        self.filenames = filenames
        self.classes = classes
"""
def centeredCrop(img, output_side_length):
    height, width, depth = img.shape
    new_height = output_side_length
    new_width = output_side_length
    if height > width:
        new_height = output_side_length * height / width
    else:
        new_width = output_side_length * width / height
    height_offset = (new_height - output_side_length) / 2
    width_offset = (new_width - output_side_length) / 2
    cropped_img = img[height_offset:height_offset + output_side_length, width_offset:width_offset + output_side_length]
    return cropped_img
"""

def preprocess(images):
    images_out = [] #final result

    """
    #Resize and crop and compute mean!
    images_cropped = []

    for i in tqdm(range(len(images))):
        X = cv2.imread(images[i])
        X = cv2.resize(X, (455, 256))
        X = centeredCrop(X, 224)
        images_cropped.append(X)
    #compute images mean
    N = 0
    mean = np.zeros((1, 3, 224, 224))
    for X in tqdm(images_cropped):
        mean[0][0] += X[:,:,0]
        mean[0][1] += X[:,:,1]
        mean[0][2] += X[:,:,2]
        N += 1
    mean[0] /= N
    #Subtract mean from all images
    for X in tqdm(images_cropped):
        X = np.transpose(X,(2,0,1))
        X = X - mean
        X = np.squeeze(X)
        X = np.transpose(X, (1,2,0))
        Y = np.expand_dims(X, axis=0)
        images_out.append(Y)
    """	

    for i in tqdm(range(len(images))):
        X = cv2.imread(images[i])
        #X = cv2.resize(X, (224, 224))
        #Y = np.expand_dims(X, axis=0)
        images_out.append(X)
    return images_out


def get_data(dataset):
    poses = []
    images = []
    filenames = []
    classes = []

    with open(directory+dataset) as f:
        next(f)  # skip the 3 header lines
        next(f)
        next(f)
        for line in f:
            fname, lat, lng, utmx, utmy, classnum = line.split()
            p0 = int(utmx)
            p1 = int(utmy)
            poses.append((p0,p1))
            images.append(directory+fname)
            filenames.append(fname)
            classes.append(classnum)
    images_out = preprocess(images)
    k = datasource(images_out, poses, filenames, classes) # list 'poses' (utm coordinates of each image) is saved as an member variable of K, which is a type of class 'datasource'

    return k
# inside the class 'datasource', there are lists 'datasource.images' and 'datasource.poses'. Each of their elements are (Width, Height, RGB channels) (e.g. (224, 224, 3)) numpy array and int (classnumber).



def getKings():
    datasource_train = get_data(dataset_train)
    datasource_test = get_data(dataset_test)

    images_train = []
    poses_train = []
    filenames_train = []
    classes_train = []

    images_test = []
    poses_test = []
    filenames_test = []
    classes_test = []

    for i in range(len(datasource_train.images)):
        images_train.append(datasource_train.images[i])
        poses_train.append((datasource_train.poses[i]))
        filenames_train.append(datasource_train.filenames[i])
        classes_train.append((datasource_train.classes[i]))

    for i in range(len(datasource_test.images)):
        images_test.append(datasource_test.images[i])
        poses_test.append((datasource_test.poses[i]))
        filenames_test.append(datasource_test.filenames[i])
        classes_test.append((datasource_test.classes[i]))
    #print "its first element: "
    #print type(datasource(images_train, poses_train).images[0])
    #print "its first element: "
    #print type(datasource(images_test, poses_test).images[0])


    return datasource(images_train, poses_train, filenames_train, classes_train), datasource(images_test, poses_test, filenames_test, classes_test)

    #print "typeof datasouce:"
    #print type(datasource(images_train, poses_train))
    #print "its member variables: "
    #print type(datasource(images_train, poses_train).images)
    #print "its first element: "
    #print type(datasource(images_train, poses_train).images[0])
    #print "its shape: "
    #print datasource(images_train, poses_train).images[0].shape

"""
typeof datasouce:
<class 'helper.datasource'>
its member variables:
<type 'list'>
its first element:
<type 'numpy.ndarray'>
its shape:
(687, 687, 3)
"""

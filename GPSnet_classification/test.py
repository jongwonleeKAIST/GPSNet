import sys
import math
import helper
import mod_vgg16
import numpy as np
import fnmatch, os, cv2
#import pylab as pl
import matplotlib.pyplot as plt

from keras.optimizers import Adam
from keras import backend as K

K.set_learning_phase(1) #set learning phase

def inliers(test_images, test_poses, pred_poses, unsuffled_csv_data_path):
    count = 0
    inliers_error = []
    for i in range(len(test_images)):
        pose = np.asarray(test_poses[i][0:2])
        pred_pose = pred_poses[i]

        error = np.linalg.norm(pose - pred_pose)

        if error < 3 * mod_vgg16.threshold(unsuffled_csv_data_path):
            count += 1
            inliers_error.append(error)

    mean_of_inliers_error = reduce(lambda x, y: x + y, inliers_error) / len(inliers_error)
    return count, mean_of_inliers_error


def unmatchedimgs(query_classnum, predicted_classnum):
    queryimg_dir = '../img_files/yuseong_ricoh/'
    trainimg_dir = '../img_files/yuseong_google/'
    for file in os.listdir(queryimg_dir): # query image
        if fnmatch.fnmatch(file ,'{}*.jpg'.format(str(query_classnum).zfill(5))):
            #print "queryimg: ", file
            img1 = cv2.imread(queryimg_dir + file)
    
    for file in os.listdir(trainimg_dir): 
        if fnmatch.fnmatch(file ,'{}*.jpg'.format(str(query_classnum).zfill(5))):
            img2 = cv2.imread(trainimg_dir + file) # correct matching
            #print "correctimg: ", file
        elif fnmatch.fnmatch(file ,'{}*.jpg'.format(str(predicted_classnum).zfill(5))):
            img3 = cv2.imread(trainimg_dir + file) # predicted matching
            #print "predictimg: ", file
    
    img1 = cv2.resize(img1, (448, 448))
    img2 = cv2.resize(img2, (448, 448))
    img3 = cv2.resize(img3, (448, 448))
    
    numpy_horizontal = np.hstack((img1, img2, img3))
    numpy_horizontal_concat = np.concatenate((img1, img2, img3), axis=1)
    
    cv2.imwrite('./180731_results/query_{}_pred_{}.jpg'.format(query_classnum, predicted_classnum), numpy_horizontal_concat)
    

def generate_cam(img_tensor, model, class_index, activation_layer):
    inp = model.input
    A_k = model.get_layer(activation_layer).output
    outp = model.layers[-1].output
    get_output = K.function([inp], [A_k, outp])
    [conv_output, predictions] = get_output([img_tensor])
    conv_output = conv_output[0]
    weightvector = model.layers[-1].get_weights()[0][:, class_index] # 1024 x 1 column vector
    cam = np.zeros(dtype=np.float32, shape=conv_output.shape[0:2])
    
    for k, w_k in enumerate(weightvector):
        cam += w_k * conv_output[:, :, k]
    
    cam = cv2.resize(cam, (224, 224))
    cam += abs(cam.min())
    cam = cam / cam.max() * 255
    cam = np.uint8(cam)
    cam = cv2.cvtColor(cv2.applyColorMap(cam, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    
    return cam


def generate_grad_cam(img_tensor, model, class_index, activation_layer):
    """
    params:
    -------
    img_tensor: image tensor
    model: pretrained model (include_top=True)
    class_index: correct class label
    activation_layer: the layer that we want to see its activation

    return:
    grad_cam: grad_cam heat map
    """
    inp = model.input
    y_c = model.output.op.inputs[0][0, class_index]  
    A_k = model.get_layer(activation_layer).output
    
    ## input: image tensor
    ## a_k: corresponding activation layer's output
    ## calculate gradient of a_k, which is an input of softmax layer.
    get_output = K.function([inp], [A_k, K.gradients(y_c, A_k)[0], model.output])
    [conv_output, grad_val, model_output] = get_output([img_tensor])

    ## reducing batch's dimension
    conv_output = conv_output[0]
    grad_val = grad_val[0]
    
    ## calculate a^c_k
    weights = np.mean(grad_val, axis=(0, 1))
    
    ## sigma(weight * conv_output) = grad_cam
    grad_cam = np.zeros(dtype=np.float32, shape=conv_output.shape[0:2])
    for k, w in enumerate(weights):
        grad_cam += w * conv_output[:, :, k]
    
    grad_cam = cv2.resize(grad_cam, (224, 224))

    ## use ReLU: make negative number to zero
    grad_cam = np.maximum(grad_cam, 0)

    grad_cam = grad_cam / grad_cam.max() * 255
    grad_cam = np.uint8(grad_cam)
    grad_cam = cv2.cvtColor(cv2.applyColorMap(grad_cam, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    
    return grad_cam


def create_heatmap(im_map, im_cloud, kernel_size=(5,5),colormap=cv2.COLORMAP_JET,a1=0.8,a2=0.2):
    '''
    img is numpy array
    kernel_size must be odd ie. (5,5)
    '''
    # create blur image, kernel must be an odd number
    im_cloud_blur = cv2.GaussianBlur(im_cloud,kernel_size,0)

    # If you need to invert the black/white data image
    # im_blur = np.invert(im_blur)
    #Convert back to BGR for cv2
    #im_cloud_blur = cv2.cvtColor(im_cloud_blur,cv2.COLOR_GRAY2BGR)

    # Apply colormap
    im_cloud_clr = cv2.applyColorMap(im_cloud_blur, colormap)

    # blend images 50/50
    return (a1*im_map + a2*im_cloud_clr).astype(np.uint8)


if __name__ == "__main__":
    seqname = sys.argv[1]
    helper.init(seqname)

    # Test model
    model = mod_vgg16.create_my_model_without_dropout()
    model.load_weights('./checkpoint_weights.h5')
    adam = Adam(lr=0.001)
    model.compile(loss={'fc_pose_utmx_utmy': mod_vgg16.euc_loss}, optimizer = adam)

    for layer in model.layers:
        layer.trainable = False 

    dataset_train, dataset_test = helper.getKings()

    X_test = np.squeeze(np.array(dataset_test.images))
    y_test = np.squeeze(np.array(dataset_test.poses))
    
    c_train = np.squeeze(np.array(dataset_train.classes))
    c_test = np.squeeze(np.array(dataset_test.classes))
    c_train = c_train.tolist()
    c_test = c_test.tolist()
    
    
    preds = model.predict(X_test)   # np array, shape of [None,2]. Predicted value is (utmx, utmy) coordinate of each test image.

    # Get results... :/
    
    unsuffled_csv_data_path = '../img_files/{}_unshuffled.csv'.format(seqname)
    print "Threshold: {}".format(3 * mod_vgg16.threshold(unsuffled_csv_data_path))
    
    error_list = np.zeros(len(dataset_test.images))
    print "sizeof errorlist : ", len(error_list)
    for i in range(len(dataset_test.images)):
        print "classes: ", c_test[i]
        pose = np.asarray(dataset_test.poses[i]) # (utmx, utmy)
        print "poses: ", pose
        pred_pose = preds[i]
        print "predicted_pose: ", pred_pose

        pose = np.squeeze(pose)
        pred_pose = np.squeeze(pred_pose)

        #Compute Individual Sample Error
        error = np.linalg.norm(pose-pred_pose)
        error_list[i] = error
        
        print 'Iteration:  ', i, '  Error latlng (m):  ', error, '\n'
    
    
    print 'Median: ', np.median(error_list), '\n'
        
    numofinliers, meanerrorofinliers = inliers(dataset_test.images, dataset_test.poses, preds, unsuffled_csv_data_path)
    print "Number of inliers: {} / {} = {} %, Mean error of inliers: {}".format(numofinliers, len(dataset_test.images), float(numofinliers)/len(dataset_test.images)*100, meanerrorofinliers)
    
    xs_test = y_test[:,0].tolist()
    ys_test = y_test[:,1].tolist()
    xs_pred = preds[:,0].tolist()
    ys_pred = preds[:,1].tolist()
    list_idx = range(len(y_test))

    
    #plt.figure(1)
    xlegend = max(xs_test + xs_pred) - min(xs_test + xs_pred)
    ylegend = max(ys_test + ys_pred) - min(ys_test + ys_pred)
    
    plt.figure(1, figsize=(xlegend/ylegend*6, 6))
    
    """
    pts_test, = plt.plot(xs_test, ys_test, "bv")
    for x, y, idx in zip(xs_test, ys_test, c_test):
        plt.text(x, y, str(idx), color="blue", fontsize=8)
    
    pts_pred, = plt.plot(xs_pred, ys_pred, "rv")
    for x, y, idx in zip(xs_pred, ys_pred, c_test):
        plt.text(x, y, str(idx), color="red", fontsize=8)
    """
    
    for i in list_idx:
        x_connects = [xs_test[i], xs_pred[i]]
        y_connects = [ys_test[i], ys_pred[i]]
        plt.plot(x_connects, y_connects, marker = None, color = 'black')
    
    
    pts_test, = plt.plot(xs_test, ys_test, "bv")
    pts_pred, = plt.plot(xs_pred, ys_pred, "rv")
    

    plt.margins(0.1)
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.title('test and pred values')
    plt.legend([pts_test, pts_pred], ["test", "pred"])
    
    
    #plt.show()
    plt.savefig('temp.png')
    

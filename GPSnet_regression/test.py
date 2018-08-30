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
    #y_test = np.reshape(y_test, (-1,1))
    
    c_train = np.squeeze(np.array(dataset_train.classes))
    c_test = np.squeeze(np.array(dataset_test.classes))
    c_train = c_train.tolist()
    c_test = c_test.tolist()
    
    
    preds = model.predict(X_test)   # np array, shape of [None,2]. Predicted value is (utmx, utmy) coordinate of each test image.

    # Get results... :/
    
    unsuffled_csv_data_path = '../img_files/{}_unshuffled.csv'.format(seqname)
    #print "Threshold: {}".format(3 * mod_vgg16.threshold(unsuffled_csv_data_path))
    
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
        
    #numofinliers, meanerrorofinliers = inliers(dataset_test.images, dataset_test.poses, preds, unsuffled_csv_data_path)
    #print "Number of inliers: {} / {} = {} %, Mean error of inliers: {}".format(numofinliers, len(dataset_test.images), float(numofinliers)/len(dataset_test.images)*100, meanerrorofinliers)
    
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
    


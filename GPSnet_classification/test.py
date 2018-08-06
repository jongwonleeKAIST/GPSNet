import sys
import math
import helper
import mod_vgg16
import numpy as np
import fnmatch, os, cv2

from keras.optimizers import Adam
from keras import backend as K

K.set_learning_phase(1) #set learning phase

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
    
    """
    img1 = cv2.resize(img1, (448, 448))
    img2 = cv2.resize(img2, (448, 448))
    img3 = cv2.resize(img3, (448, 448))
    
    numpy_horizontal = np.hstack((img1, img2, img3))
    numpy_horizontal_concat = np.concatenate((img1, img2, img3), axis=1)
    
    cv2.imwrite('./180731_results/query_{}_pred_{}.jpg'.format(query_classnum, predicted_classnum), numpy_horizontal_concat)
    """


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
    model.load_weights('./trained_weights.h5')
    adam = Adam(lr=0.0001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer = adam, metrics=['accuracy'])

    for layer in model.layers:
        layer.trainable = False


    dataset_train, dataset_test = helper.getKings()

    X_test = np.squeeze(np.array(dataset_test.images))
    y_test = np.squeeze(np.array(dataset_test.imgclasses))
    y_test = np.reshape(y_test, (-1,1))
    
    predictions_to_return = 5
    
    top1idx = 0
    top5idx = 0
    
    for i in range(len(y_test)):
        preds = model.predict(X_test)[i] # list of predicted probablity for each class
        
        top_preds = np.argsort(preds)[::-1][0:predictions_to_return] # list of sorted (higher to lower) index of probability
        
        top_preds_prob = []
        for j in range(predictions_to_return):
            top_preds_prob.append((top_preds[j], preds[top_preds[j]]))
    
        print "y_test: {}, top_preds: {}, top5: {}\ntop_preds_prob: {}".format(y_test[i], top_preds, y_test[i] in top_preds, top_preds_prob)
        if y_test[i] == top_preds[0]:
            top1idx += 1.
            
            #im_map = X_test[i]
            """
            for k in range(5):
                blocknum = k + 1 
                im_cloud = generate_grad_cam(X_test[i].reshape(1,224,224,3), model, y_test[i], 'block{}_conv1'.format(blocknum))
                im_heatmap = create_heatmap(im_map, im_cloud)
                cv2.imwrite('./180731_results/result_gradCAM/results_for_each_block/class{}_block{}_conv1.jpg'.format(y_test[i][0], blocknum), im_heatmap)
            
            im_cloud = generate_grad_cam(X_test[i].reshape(1,224,224,3), model, y_test[i], 'block5_conv3')
            im_heatmap = create_heatmap(im_map, im_cloud)
            cv2.imwrite('./180731_results/result_gradCAM/results_for_each_block/class{}_block5_conv3.jpg'.format(y_test[i][0]), im_heatmap)
            """    
                
        if y_test[i] in top_preds:
            top5idx += 1.
        else:
            print "Unmatched {} and {}".format(y_test[i], top_preds[0])
            #im_map = X_test[i]
            #im_cloud = generate_grad_cam(X_test[i].reshape(1,224,224,3), model, top_preds[0], 'block5_conv3')
            #im_heatmap = create_heatmap(im_map, im_cloud)     
            #cv2.imwrite('./180731_results/query_{}_predicted_{}.jpg'.format(y_test[i][0], top_preds[0]), im_heatmap)
            unmatchedimgs(int(y_test[i]), top_preds[0])
    
    top1accuracy = top1idx / len(y_test)
    top5accuracy = top5idx / len(y_test)
    
    print "Top 1: {}, Top 5: {}".format(top1accuracy, top5accuracy)
        
    

    """
    vals = testPredict # np array, shape of [None,2]. Predicted value is (utmx, utmy) coordinate of each test image.

    # Get results... :/
    unsuffled_csv_data_path = './img_files/{}_unshuffled.csv'.format(seqname)
    
    results = np.zeros(len(dataset_test.images))
    for i in range(len(dataset_test.images)):

        imgclasses = np.asarray(dataset_test.imgclasses[i]) # (utmx, utmy)
        print "imgclasses: ", imgclasses
        pred_imgclasses = vals[i]
        print "predicted_imgclasses: ", imgclasses
    """

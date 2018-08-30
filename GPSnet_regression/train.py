import sys
import helper
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import Adam, SGD
from keras import metrics, losses
from keras.utils import np_utils

import matplotlib.pyplot as plt

import mod_vgg16


if __name__ == "__main__":
    seqname = sys.argv[1]
    helper.init(seqname)
    
    
    # Variables
    
    learning_rate = float(sys.argv[2])
    
    batch_size = 75
    epochs = 100
    #learning_rate = 1e-6

    # Train model
    model = mod_vgg16.create_my_model() 
    adam = Adam(lr = learning_rate, clipvalue=2) #decay = 1e-5
    model.compile(loss={'fc_pose_utmx_utmy': mod_vgg16.euc_loss}, optimizer = adam)

    dataset_train, dataset_test = helper.getKings()
    
    # get 'helper.datasource' type datasets; its images and poses compoments are stored as 'list' type, and images are numpy array size of (224, 224,3) and poses are tuple of (utmx, utmy)
    X_train = np.squeeze(np.array(dataset_train.images))
    y_train = np.squeeze(np.array(dataset_train.poses))
    
    X_test = np.squeeze(np.array(dataset_test.images))
    y_test = np.squeeze(np.array(dataset_test.poses))
    
    print "X_train.shape", X_train.shape
    print "y_train.shape", y_train.shape
    
    print "X_test.shape", X_test.shape
    print "y_test.shape", y_test.shape
    
    # Setup checkpointing
    checkpointer = ModelCheckpoint(filepath="./checkpoint_weights.h5", verbose=1, save_best_only=True, save_weights_only = True)
    es = EarlyStopping(patience=5, monitor='val_loss')
    tb = TensorBoard(log_dir='vgg16_tl')
    
    
    history = model.fit(X_train, y_train,
          batch_size = batch_size,
          epochs = epochs,
          validation_data=(X_test, y_test),
          callbacks=[checkpointer,es, tb])

    
    print(history.history.keys())
    plt.figure(1)
    
    # summarize history for loss  

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    plt.savefig("./training_graph.png")
    
    model.save_weights("./trained_weights.h5")


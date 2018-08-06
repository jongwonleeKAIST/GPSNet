import sys
import helper
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import Adam
from keras import metrics
from keras.utils import np_utils

import matplotlib.pyplot as plt

import mod_vgg16


if __name__ == "__main__":
    seqname = sys.argv[1]
    helper.init(seqname)
    
    
    # Variables
    
    learning_rate = float(sys.argv[2])
    
    batch_size = 30
    epochs = 20
    #learning_rate = 1e-6

    # Train model
    model = mod_vgg16.create_my_model()
    adam = Adam(lr = learning_rate, clipvalue = 1.5)
    model.compile(loss='sparse_categorical_crossentropy', optimizer = adam, metrics=['accuracy'])
#metrics = [metrics.sparse_categorical_accuracy]
    dataset_train, dataset_test = helper.getKings()
    
    print "dataset_train, dataset_test : ", type(dataset_train.images[0]), type(dataset_test.images[0])
    
    # get 'helper.datasource' type datasets; its images and poses compoments are stored as 'list' type, and images are numpy array size of (448, 448,3) and poses are tuple of (utmx, utmy)
    X_train = np.squeeze(np.array(dataset_train.images))
    y_train = np.squeeze(np.array(dataset_train.imgclasses))
    y_train = np.reshape(y_train, (-1,1))
    
    X_test = np.squeeze(np.array(dataset_test.images))
    y_test = np.squeeze(np.array(dataset_test.imgclasses))
    y_test = np.reshape(y_test, (-1,1))

    # Setup checkpointing
    checkpointer = ModelCheckpoint(filepath="./checkpoint_weights.h5".format(seqname, epochs, learning_rate), verbose=1, save_best_only=True, save_weights_only = True)
    es = EarlyStopping(patience=5, monitor='val_acc')
    tb = TensorBoard(log_dir='vgg16_tl')
    
    
    history = model.fit(X_train, y_train,
          batch_size = batch_size,
          epochs = epochs,
          validation_data=(X_test, y_test),
          callbacks=[checkpointer, es, tb])

    
    print(history.history.keys())
    plt.figure(1)
    
    # summarize history for loss  

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    plt.savefig("./training_graph.png".format(seqname, epochs, learning_rate))
    
    model.save_weights("./trained_weights.h5".format(seqname, epochs, learning_rate))

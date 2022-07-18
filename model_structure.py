'''
K-Fold Cross-Validation, training method and model structure.
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense,Conv1D,MaxPooling1D,GlobalAveragePooling1D,Dropout
from tensorflow.keras.models import Sequential
import os
from tool import mkdir
import matplotlib.pylab as plt


def train_test_valid_set(x,y):
        
    x=np.vsplit(x,10)
    y=np.vsplit(y,10)
    
    xz=np.zeros(384)[:,np.newaxis].T
    yz=np.zeros(int(len(y)/10))[:,np.newaxis]
    
    #test
    x_test=xz
    for i in range(5,10):
        x_test=np.vstack((x_test,x[i]))
    x_test=np.delete(x_test,0,0)
    
    y_test=yz
    for i in range(4,9):
        y_test=np.vstack((y_test,y[i]))
    for i in range(int(len(y)/10)):
        y_test=np.delete(y_test,0,0)
    
    #valid
    x_valid=xz
    for i in range(5,10):
        x_valid=np.vstack((x_valid,x[i]))
    x_valid=np.delete(x_valid,0,0)
    
    y_valid=yz
    for i in range(5,10):
        y_valid=np.vstack((y_valid,y[i]))
    for i in range(int(len(y)/10)):
        y_valid=np.delete(y_valid,0,0)
        
    #train
    x_train=xz
    for k in range(4,9):
        list10=[j for j in range(10)]
        list10.remove(k)
        list10.remove(k+1)
        data=xz
        for z in list10:
            data=np.vstack((data,x[z]))
        data=np.delete(data,0,0)
        x_train=np.vstack((x_train,data))
    x_train=np.delete(x_train,0,0)
    
    y_train=yz
    for k in range(4,9):
        list10=[j for j in range(10)]
        list10.remove(k)
        list10.remove(k+1)
        data=yz
        for z in list10:
            data=np.vstack((data,y[z]))
        for i in range(int(len(y)/10)):
            data=np.delete(data,0,0)
        y_train=np.vstack((y_train,data))
    for i in range(int(len(y)/10)):
        y_train=np.delete(y_train,0,0)
    

    kfold_num=5
    x_train=np.vsplit(x_train,kfold_num)
    y_train=np.vsplit(y_train,kfold_num)
    x_valid=np.vsplit(x_valid,kfold_num)
    y_valid=np.vsplit(y_valid,kfold_num)
    x_test=np.vsplit(x_test,kfold_num)
    y_test=np.vsplit(y_test,kfold_num)
    
    for i in range(kfold_num):
        
        x_train[i],y_train[i]=data_expand_dim(x_train[i], y_train[i])
        x_valid[i],y_valid[i]=data_expand_dim(x_valid[i], y_valid[i])
        x_test[i],y_test[i]=data_expand_dim(x_test[i], y_test[i])
    
    return x_train,x_test,x_valid,y_train,y_test,y_valid

def data_expand_dim(x,y): 

    x=np.expand_dims(x,axis=2)
    #y=y[:,np.newaxis]
    
    return x,y

def conv_1d(x_train,x_test,x_valid,y_train,y_test,y_valid,path,N):
    #define per fold score containers
    train_acc_per_fold=[]
    acc_per_fold=[]
    epochs_per_fold=[]

    count=1
    for i in range(5):

        model=Sequential()
        model.add(Conv1D(64, 5,strides=1, activation='relu',input_shape=(384, 1)))
        model.add(Conv1D(64, 5, strides=1,activation='relu'))
        model.add(MaxPooling1D(3))
        model.add(Conv1D(64, 5, strides=1,activation='relu'))
        model.add(Conv1D(64, 5, strides=1,activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        epochs_count=0
        finish_step=0
        diff_acc=1.0
        old_acc=0.0
        while (diff_acc>1e-3 or finish_step<3):
                        
            history=model.fit(x_train[i], y_train[i], batch_size=64,epochs=1, verbose=0,validation_data=(x_valid[i],y_valid[i]))
            val_accuracy=history.history['val_accuracy']
            #diff_acc=np.abs(val_accuracy[0]-old_acc)
            diff_acc=(val_accuracy[0]-old_acc)
            if diff_acc<1e-3:
                finish_step+=1
            else:
                finish_step=0
            #print('finish_step:',finish_step)
            old_acc=val_accuracy[0]
            epochs_count+=1
            #print('epochs=',epochs_count)
            #print('old_acc=',old_acc)

        train_acc_per_fold.append(history.history['accuracy'][0])
        evaluate=model.evaluate(x_test[i],y_test[i])
        print('N: ',str(N),'- fold',count,':','train acc: ',history.history['accuracy'][-1],' test acc: ', evaluate[1])
        acc_per_fold.append(evaluate[1])
        epochs_per_fold.append(epochs_count)

        model.save(os.path.join(path,os.path.join(str(N)+'-'+str(count)+'.h5')))

        count=count+1

    np.savetxt(os.path.join(path,str(N)+'train_record.txt'),np.vstack((train_acc_per_fold,acc_per_fold,epochs_per_fold)),fmt='%10.4f')

    return

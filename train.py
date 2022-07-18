'''
train our model and record training time.
'''
import numpy as np
from model_structure import train_test_valid_set,conv_1d
from tool import mkdir
import time
import os

#data放置資料夾
datapath=os.path.join(os.getcwd(),'train_data')
kplr_datapath=os.path.join(datapath,'kepler')
pearson_datapath=os.path.join(datapath,'pearson')

#result放置資料夾
result_folder=os.path.join(os.getcwd(),'train_result')
mkdir(result_folder)
kplr_result_folder=os.path.join(result_folder,'kepler')
mkdir(kplr_result_folder)
pearson_result_folder=os.path.join(result_folder,'pearson')
mkdir(pearson_result_folder)


#Ns=[2000,4000,8000,12000,16000,20000]
Ns=[16000]
#kepler
k_cost_time_record=[]

for i in range(len(Ns)):
    start_time = time.time()
    
    N=Ns[i]

    x=np.load(os.path.join(kplr_datapath,'flux_k_'+str(N)+'.npy'),allow_pickle=True)
    x-=1
    y=np.load(os.path.join(kplr_datapath,'label_k_'+str(N)+'.npy'),allow_pickle=True)
    y=y[:,np.newaxis]
    
    x_train,x_test,x_valid,y_train,y_test,y_valid=train_test_valid_set(x,y)
    conv_1d(x_train,x_test,x_valid,y_train,y_test,y_valid,kplr_result_folder,N)
    end_time=time.time()
    k_cost_time_record.append([N,end_time-start_time])
    print(end_time-start_time)
np.savetxt(os.path.join(kplr_result_folder,'cost_time_record.txt'),k_cost_time_record,fmt='%10.3f')
'''
#pearson
p_cost_time_record=[]
for i in range(len(Ns)):
    start_time = time.time()
    
    N=Ns[i]

    x=np.load(os.path.join(pearson_datapath,'flux_p_'+str(N)+'.npy'),allow_pickle=True)
    x-=1
    y=np.load(os.path.join(pearson_datapath,'label_p_'+str(N)+'.npy'),allow_pickle=True)
    y=y[:,np.newaxis]
    
    x_train,x_test,x_valid,y_train,y_test,y_valid=train_test_valid_set(x,y)
    conv_1d(x_train,x_test,x_valid,y_train,y_test,y_valid,pearson_result_folder,N)
    end_time=time.time()
    p_cost_time_record.append([N,end_time-start_time])
    print(end_time-start_time)
np.savetxt(os.path.join(pearson_result_folder,'cost_time_record.txt'),p_cost_time_record,fmt='%10.3f')
'''
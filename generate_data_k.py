'''
to generate k dataset.
'''

import numpy as np
from tool import ha_z,fold,interpolation,mkdir
from mangol import mandel_agol as mangol
import time
import os
import matplotlib.pylab as plt

np.random.seed(1)
s=time.time()
def save_record(t,flux,label,rp_rs,a_rs,iang,t0,khap,sigma):
    time_record.extend([t])
    LC_record.extend([flux])
    label_record.append(np.float64(label))
    parameter_record.append([rp_rs, a_rs, iang, t0, khap, sigma])

path=os.getcwd()
folder=os.path.join(path,'train_data')
mkdir(folder)
folder=os.path.join(folder,'kepler')
mkdir(folder)

noisepath=os.path.join(os.getcwd(),'Kplr')
timebar=np.load(os.path.join(noisepath,'kepler_data_based_time.npy'),allow_pickle=True)
fluxbar=np.load(os.path.join(noisepath,'kepler_data_based_flux.npy'),allow_pickle=True)
id=np.loadtxt(os.path.join(noisepath,'kepler_data_based_filename.txt'),unpack=True)


Ns=[2000,4000,8000,12000,16000,20000]
for i in range(len(Ns)):

    s=time.time()
    N=Ns[i]
    time_record = []
    LC_record = []
    label_record = []
    parameter_record = []
    nan_record=[]
    for j in range(N):
        r=np.random.choice(len(timebar))
        
        t=timebar[r]
        flux=fluxbar[r]
        file=id[r]
        
        u1 = 0.5  # linear limb darkening term
        u2 = 0  # quadratic
        khap=np.random.uniform(2,4)#period
        rp_rs = np.random.uniform(0.06,0.1)
        a_rs = np.random.uniform(8,35)
        iang = np.random.uniform(86,90)
        t0 = np.random.uniform(0.1,khap-0.3)

        z = ha_z(t,t0,khap,a_rs,iang)
        y = mangol(z,u1,u2,rp_rs)
        yk = y*flux
        yk_noise = flux
        if np.isnan(yk).sum()>0:
            nan_record.append(r)
            print('nan is in ',r)

        if np.isnan(yk_noise).sum()>0:
            nan_record.append(r)
            print('nan is in ',r)

            
        tfold=[khap,khap+2/1440,khap-2/1440]
        for k in range(len(tfold)):
            #non-transit
            noise_time,noise_flux,interval_flux=fold(tfold[k],t,yk_noise,15.0)
            noise_time, noise_flux =interpolation(noise_time,noise_flux,tfold[k],384)
            if np.isnan(noise_flux).sum()>0:
                print(r,'th. nan is on fold ',k)
                false_flux=noise_flux
                false_interval_flux=interval_flux
            
                plt.plot(noise_time,noise_flux,'b-')
                plt.title(str(r))
                plt.show()
            save_record(noise_time,noise_flux,0,rp_rs, a_rs, iang, t0, khap, file)

            #transit
            signal_time,signal_flux,_=fold(tfold[k],t,yk,15.0)            
            signal_time, signal_flux =interpolation(signal_time,signal_flux,tfold[k],384)
            
            save_record(signal_time,signal_flux,1,rp_rs, a_rs, iang, t0, khap, np.std(noise_flux))


    np.save(os.path.join(folder,'time_k_'+str(N)+'.npy'),time_record)#10.3f=10進制+小數點後三位float
    np.save(os.path.join(folder,'flux_k_'+str(N)+'.npy'),LC_record)#10.5f=10進制+小數點後五位float
    np.save(os.path.join(folder,'label_k_'+str(N)+'.npy'),label_record)#i=integer
    np.save(os.path.join(folder,'parameter_k_'+str(N)+'.npy'),parameter_record)
    
    e=time.time()
    print('cost ',e-s)

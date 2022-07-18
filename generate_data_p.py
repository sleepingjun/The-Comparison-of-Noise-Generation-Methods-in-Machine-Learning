'''
to generate p dataset.
'''
import numpy as np
import random
from tool import ha_z,mkdir
from mangol import mandel_agol as mangol
import time
import os

def save_record(t,flux,label,rp_rs,a_rs,iang,t0,khap,A):
    time_record.extend([t])
    LC_record.extend([flux])
    label_record.append(np.float64(label))
    parameter_record.append([rp_rs, a_rs, iang, t0, khap, A])

def person_generatemethod(time_input, transit_function, A,rp_rs):
    sig_tol = np.linspace(*[1.5, 4.5, 4])
    phi = 0
    wave_period = [6.0 / 24, 12.0 / 24, 24.0 / 24]
    amplitude_variability_period = np.asarray([-1, 1, 100])
    wave_variability_period = np.asarray([-3, 1, 100])

    w=random.choice(wave_period)
    PA=random.choice(amplitude_variability_period)
    PW=random.choice(wave_variability_period)

    # compute white noise
    sig=random.choice(sig_tol)
    noise = rp_rs ** 2 / sig
    gdist = np.random.normal(1, noise, len(time_input))  # Gaussian(normal) distribution

    # generate transit + variability
    t = time_input - np.min(time_input)
    A = A + A * np.sin(2 * np.pi * t / PA)
    ww = w + w * np.sin(2 * np.pi * t / PW)
    data = transit_function * gdist * (1 + A * np.sin(2 * np.pi * t / ww + phi))  # transit's noisy
    ndata = gdist * (1 + A * np.sin(2 * np.pi * t / ww + phi))  # non-transit's noisy
    param=[w,PA,PW,sig]
    return t,data, ndata,param

path=os.getcwd()
folder=os.path.join(path,'train_data')
mkdir(folder)
folder=os.path.join(folder,'pearson')
mkdir(folder)

np.random.seed(1)

Ns=[2000,4000,8000,12000,16000,20000]
for i in range(len(Ns)):
    s=time.time()
    N = Ns[i]*3
    
    time_record = []
    LC_record = []
    label_record = []
    parameter_record = []

    for j in range(N):

        #set parameters
        u1 = 0.5  # linear limb darkening term
        u2 = 0  # quadratic
        khap=np.random.uniform(2,4)#period
        t=np.linspace(*[0, khap, 384])
        rp_rs =np.random.uniform(0.06,0.1)
        a_rs = np.random.uniform(5,35)
        iang = np.random.uniform(86,90)
        t0 = np.random.uniform(0.1,khap-0.3)
        #generate transit
        zz = ha_z(t,t0,khap,a_rs,iang)
        transit_function = mangol(zz,u1,u2,rp_rs)

        a = [250, 500, 1000, 2000]
        bd=N/len(a)
        if j<bd:
            A = a[0]*1e-6
        elif bd<=j<bd*2:
            A = a[1] * 1e-6
        elif bd*2<=j<bd*3:
            A = a[2] * 1e-6
        elif bd*3<=j<bd*4:
            A = a[3] * 1e-6

        t_p,data,ndata,p_param=person_generatemethod(t, transit_function, A,rp_rs)

        # non-transit
        save_record(t_p,ndata,0, rp_rs, a_rs, iang, t0, khap, A)
        # transit
        save_record(t_p,data,1,rp_rs, a_rs, iang, t0, khap,A)

    np.save(os.path.join(folder,'time_p_'+str(Ns[i])+'.npy'),time_record)
    np.save(os.path.join(folder,'flux_p_'+str(Ns[i])+'.npy'),LC_record)
    np.save(os.path.join(folder,'label_p_'+str(Ns[i])+'.npy'),label_record)
    np.save(os.path.join(folder,'parameter_p_'+str(Ns[i])+'.npy'),parameter_record)

e=time.time()
print('cost ',e-s,', total data: ',len(label_record))
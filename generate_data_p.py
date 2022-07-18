import numpy as np
import random
from tool import ha_z,mkdir
from mangol import mandel_agol as mangol #limb darkening
import time
import os
import matplotlib.pylab as plt

def save_record(t,flux,label,rp_rs,a_rs,iang,t0,khap,A):                    #儲存(save)的函數
    time_record.extend([t])
    LC_record.extend([flux])
    label_record.append(np.float64(label))
    parameter_record.append([rp_rs, a_rs, iang, t0, khap, A])

def person_generatemethod(time_input, transit_function, A,rp_rs):
    sig_tol = np.linspace(*[1.5, 4.5, 4])  # use 1.5
    phi = 0
    wave_period = [6.0 / 24, 12.0 / 24, 24.0 / 24]  # wave period
    amplitude_variability_period = np.asarray([-1, 1, 100])  # use 100
    wave_variability_period = np.asarray([-3, 1, 100])  # use 100

    w=random.choice(wave_period)
    PA=random.choice(amplitude_variability_period)
    PW=random.choice(wave_variability_period)
    '''
    w=wave_period[1]
    PA=amplitude_variability_period[1]
    PW=wave_variability_period[1]'''

    # compute white noise
    sig=random.choice(sig_tol)
    noise = rp_rs ** 2 / sig
    #noise=rp_rs ** 2 / (sig_tol)
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

Ns=[16000]
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
        #t=khap
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

        '''print(A)
        print(p_param)'''
        # non-transit
        save_record(t_p,ndata,0, rp_rs, a_rs, iang, t0, khap, A)
        # transit
        save_record(t_p,data,1,rp_rs, a_rs, iang, t0, khap,A)
        '''print(rp_rs)
        print(a_rs)
        print(iang)
        print(t0)'''
    print(len(LC_record))
    np.save(os.path.join(folder,'time_p_'+str(Ns[i])+'.npy'),time_record)
    np.save(os.path.join(folder,'flux_p_'+str(Ns[i])+'.npy'),LC_record)
    np.save(os.path.join(folder,'label_p_'+str(Ns[i])+'.npy'),label_record)
    np.save(os.path.join(folder,'parameter_p_'+str(Ns[i])+'.npy'),parameter_record)

e=time.time()
print('cost ',e-s,', total data: ',len(label_record))
'''

#parameter compare fig
import matplotlib.pylab as plt

u1 = 0.5;u2 = 0;
khap = np.random.uniform(2, 4)
t = np.linspace(*[0, khap, 384])
rp_rs = np.random.uniform(0.06, 0.1)
a_rs = np.random.uniform(5, 35)
iang = np.random.uniform(86, 90)
t0 = np.random.uniform(0.1, khap - 0.3)
# generate transit
zz = ha_z(t, t0, khap, a_rs, iang)
transit_function = mangol(zz, u1, u2, rp_rs)

A=250*1e-6;sig_tol=np.array(1.5);
t1, _, n1 = person_generatemethod(t, transit_function, A, rp_rs,sig_tol)
A=2000*1e-6;sig_tol=np.array(1.5);
t2, _, n2 = person_generatemethod(t, transit_function, A, rp_rs,sig_tol)
A=250*1e-6;sig_tol=np.array(4.5);
t3, _, n3 = person_generatemethod(t, transit_function, A, rp_rs,sig_tol)
A=2000*1e-6;sig_tol=np.array(4.5);
t4, _, n4 = person_generatemethod(t, transit_function, A, rp_rs,sig_tol)

fig,ax=plt.subplots(2,2,sharey=True)
ax[0,0].plot(t1,n1,'b-',label='A=250ppm, $\sigma_{tol}=1.5$')
ax[0,0].legend(loc='upper left')
ax[0,0].text(0.97, 0.05, '(a)', size=10,
               horizontalalignment='center', verticalalignment='center', transform=ax[0,0].transAxes)
ax[0,0].set_ylabel('flux')
ax[0,1].plot(t2,n2,'b-',label='A=2000ppm, $\sigma_{tol}=1.5$')
ax[0,1].legend(loc='upper left')
ax[0,1].text(0.97, 0.05, '(b)', size=10,
               horizontalalignment='center', verticalalignment='center', transform=ax[0,1].transAxes)
ax[1,0].plot(t3,n3,'b-',label='A=250ppm, $\sigma_{tol}=4.5$')
ax[1,0].legend(loc='upper left')
ax[1,0].text(0.97, 0.05, '(c)', size=10,
               horizontalalignment='center', verticalalignment='center', transform=ax[1,0].transAxes)
ax[1,0].set_ylabel('flux')
ax[1,0].set_xlabel('t')
ax[1,1].plot(t4,n4,'b-',label='A=2000ppm, $\sigma_{tol}=4.5$')
ax[1,1].legend(loc='upper left')
ax[1,1].text(0.97, 0.05, '(d)', size=10,
               horizontalalignment='center', verticalalignment='center', transform=ax[1,1].transAxes)
ax[1,1].set_xlabel('t')
fig.tight_layout()
fig.subplots_adjust(hspace=0,wspace=0)
plt.savefig('pearson_parameter_influence.png')
plt.show()
'''

train_data_folder=os.path.join(os.getcwd(),'train_data')
k_data_folder=os.path.join(train_data_folder,'kepler')
kt,kf=np.load(os.path.join(k_data_folder,'time_k_16000.npy'),allow_pickle=True),np.load(os.path.join(k_data_folder,'flux_k_16000.npy'),allow_pickle=True)

for k_num in range(16000):
    if k_num==6048:
        fig, ax = plt.subplots(1, 2, sharey=True)

        ax[0].plot(kt[k_num], kf[k_num], 'steelblue', label='noise')
        ax[0].plot(kt[k_num+1], kf[k_num + 1], 'sandybrown', label='transit')
        ax[0].set_title('kepler'+str(k_num))
        ax[0].legend()
        ax[0].set_ylabel('flux')
        ax[0].set_xlabel('time(days)')


        ax[1].plot(t_p,ndata, 'steelblue', label='noise')
        ax[1].plot(t_p,data, 'sandybrown', label='transit')
        ax[1].set_title('pearson')
        ax[1].legend(loc='lower right')
        ax[1].set_xlabel('time(days)')


        fig.tight_layout()
        fig.subplots_adjust(wspace=0,hspace=0)
        plt.close()
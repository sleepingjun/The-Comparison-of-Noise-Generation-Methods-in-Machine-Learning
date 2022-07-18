'''
commom tools
'''
import numpy as np
import math
import os
import random
import matplotlib.pyplot as plt

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('-----create successfully-----')

def ha_z(t,t0,khap,a_rs,iang):
    phi = (t-t0)/khap*2*np.pi
    z = a_rs*(np.sin(phi)**2+(np.cos(np.radians(iang))*np.cos(phi))**2)
    z[(phi/(2*np.pi)-0.25)%1<0.5] = z.max()
    return z


def fold(tfold,tmid,flux,timestep):
    tmid0=[tmid[i]-tmid[0] for i in range(len(tmid))]
    Td = np.array(tmid0) % tfold
    interval = timestep / 1440.0
    tn = math.ceil(tfold / interval)
    interval_t=[interval*i for i in range(tn)]
    interval_flux = [[] for i in range(tn)]
    average_flux = []

    for j in range(len(interval_t)):
        a=interval_t[j]
        b=(j+1)*interval
        candidate_index=np.where((Td>=a) & (Td<=b))[0]
        for i in candidate_index:
            interval_flux[j].append(flux[i])

    for i in range(len(interval_flux)):
        average_flux.append(np.mean(interval_flux[i]))

    interval_t = [(timestep / 2) / 1440.0 + interval * i for i in range(tn)]

    return interval_t, average_flux,interval_flux

def interpolation(t,flux,tfold,new_interval_num):

    delta_t0=tfold/len(t)
    delta_t = tfold/new_interval_num

    interval_it=[(delta_t/2)+delta_t*i for i in range(new_interval_num)]

    new_iflux=[0 for i in range(new_interval_num)]

    temp=0
    for i in range(1,len(t)):
        for j in range(temp,new_interval_num):
            if (0<interval_it[j]<t[0]):
                new_iflux[j] = 1 + ((interval_it[j]) / (delta_t0/2)) * (flux[i-1] - 1)
                continue
            elif(t[i-1]<interval_it[j]<t[i]):
                new_iflux[j]=flux[i-1]+((interval_it[j]-t[i-1])/delta_t0)*(flux[i]-flux[i-1])
                continue
            elif(t[-1]<interval_it[j]):
                new_iflux[j]=flux[-1]+(((interval_it[-1] - (t[-1]+delta_t0/2)) / delta_t0) * ( 1- flux[-1]))
                continue
            elif(interval_it[j]>t[i-1] and interval_it[j]>t[i]):
                temp=j
                break
    
    if np.isnan(new_iflux).sum()>0:
        sigma=np.nanstd(new_iflux)
        mean=np.nanmean(new_iflux)
        n=2
        random_num=[]
    
        for i in range(len(new_iflux)):
            if (mean-n*sigma)<=new_iflux[i]<=(mean+n*sigma):
                random_num.append(new_iflux[i])
    
        n=np.where(np.isnan(new_iflux))
        new_iflux=list(new_iflux)
        for i in range(len(n[0])):
            new_iflux[n[0][i]]=random.choice(random_num)

    return interval_it,new_iflux

def parameter(index,param):
    #param=np.loadtxt('parameter.txt',unpack=True)
    light_curve_parameter=param[index]
    print('The parameters of',index,'light curve is :')
    print('The ratio of planet radius to the stellar radius (rp_rs):',light_curve_parameter[0])
    print('The ratio of the orbital semi-major axis to the stellar radius (a_rs):',light_curve_parameter[1])
    print('Theta (iang):',light_curve_parameter[2])
    print('The mid-transit time (t0):',light_curve_parameter[3])
    print('The transit period (khap):',light_curve_parameter[4])
    print('noise sigma/id/A:',light_curve_parameter[5])

#result analysis tool

def pred_error_record(x,path, checklist, N):
    # label是1;預測是0
    prediction_error_1to0 = checklist.loc[(checklist['label'] == 1) & (checklist['prediction'] == 0)]
    prediction_error_1to0.to_csv(os.path.join(path, '1to0.csv'))
    errorlist_1to0 = prediction_error_1to0.index.tolist()
    #print('1to0 : ', len(errorlist_1to0))
    # label是0;預測是1
    prediction_error_0to1 = checklist.loc[(checklist['label'] == 0) & (checklist['prediction'] == 1)]
    prediction_error_0to1.to_csv(os.path.join(path, '0to1.csv'))
    errorlist_0to1 = prediction_error_0to1.index.tolist()
    #print('0to1 : ', len(errorlist_0to1))

    # draw error pictures
    '''imgpath10 = os.path.join(path, '1to0')
    mkdir(imgpath10)
    for i in errorlist_1to0:
        figurename = str(i) + ".png"
        img = draw_save_picture(x[i, :], imgpath10, figurename)

    imgpath01 = os.path.join(path, '0to1')
    mkdir(imgpath01)
    for i in errorlist_0to1:
        figurename = str(i) + ".png"
        img = draw_save_picture(x[i, :], imgpath01, figurename)'''

def draw_save_picture(data, path, figurename):
    plt.plot(data, 'b-', markersize=1)
    plt.savefig(os.path.join(path, figurename))
    plt.close()

def error_param_histgram(column_name,xlabel,path):
    plt.hist(column_name)
    plt.xlabel(xlabel)
    plt.ylabel('N')
    plt.savefig(path+'error_param_record_histgram-'+xlabel)
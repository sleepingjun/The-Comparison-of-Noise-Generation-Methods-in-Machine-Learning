'''
pre-processing datas from selected kepler Q1 dataset.
plot the process.
'''

import numpy as np
import math
import random
import matplotlib.pylab as plt
import os
import time

def delnan(data):
    '''
    to delete nan value from the original data.
    '''
    t=list(data[0])
    f=list(data[1])
    for i in range(len(f)):
        if np.isnan(f).sum()>0:
            n=np.where(np.isnan(f))
            t.pop(n[0][0])
            f.pop(n[0][0])

            continue
        else:
            break

    return t,f

def predeal(tmid,flux,time_step):
    '''time interval'''
    time_step=15.0
    tmid=tmid-np.amin(tmid)
    t_index=len(tmid)
    dt_half=(time_step/2)/1440.0
    interval_length=time_step/1440.0
    if tmid[-1]%interval_length==0:
        interval_num=int(tmid[-1]/interval_length)
    else:
        interval_num=int(tmid[-1]/interval_length)+1
    t=np.array([interval_length*i for i in range(interval_num+1)])

    np_out=np.zeros(interval_num)
    f_record=np.full([interval_num],np.nan)
    for i in range(0,interval_num):
        index_oldtime=[]
        index_newtime=[]
        for k in range(len(tmid)):
            if(tmid[k]>=t[i] and tmid[k]<=t[i+1]):
                np_out[i]=np_out[i]+1.0
                index_oldtime.append(k)
                index_newtime.append(i)
            else:
                continue
            
        for a in index_oldtime:
            for b in index_newtime:
                record=[]
                record.append(flux[a])
                f_record[b]=np.sum(record)/np_out[b]               
                
    #step2. divided into several groups
    left=[]
    right=[]
        #群左包開頭是否有點的情況
    if np_out[0]!=0:#index 0 是左包
        left.append(0)
        for j in range(1,len(np_out)):
            if(np_out[j]!=0 and np_out[j-1]==0):
                left.append(j)
    else:
        for j in range(1,len(np_out)):
            if(np_out[j]!=0 and np_out[j-1]==0):
                left.append(j)
        #群右包最後一個的情況
    for i in range(len(np_out)-1):
        if(np_out[i]!=0 and np_out[i+1]==0):
            right.append(i)    
    if np_out[-1]!=0:
        right.append(len(np_out))
        
    a=[left[0]]
    b=[]
    for k in range(1,len(left)):
        if left[k]-right[k-1]>=(400/time_step):
            a.append(left[k])
            b.append(right[k-1])
    b.append(right[-1])

    #data is cont.
    if len(a)==1:
        f_record[:]=f_record[:]/np.nanmean(f_record[:])
        middle_t=np.array([dt_half+interval_length*i for i in range(interval_num)])#區間間隔t
    else:
        print('this file is not cont.')

    return middle_t,f_record

def minus_dev(t,f,n):
    '''
    minus n times dev
    '''
    sigma=np.nanstd(f)
    mean=np.nanmean(f)
    
    f1=[]
    t1=[]

    for i in range(len(t)):
        if (mean-n*sigma)<=f[i]<=(mean+n*sigma):
            f1.append(f[i])
            t1.append(t[i])

    return t1,f1

def fold(tfold,tmid,flux,timestep):

    Td=np.array(tmid)%tfold
    interval=timestep/1440.0
    tn=math.ceil(tfold/interval)
    interval_t=[interval*i for i in range(tn)]
    interval_flux=[[]for i in range(tn)]
    average_flux=[]

    for i in range(len(Td)):
        for j in range(len(interval_t)):
            a=interval_t[j]
            b=(j+1)*interval
            if(Td[i]<b and Td[i]>=a):
                interval_flux[j].append(flux[i])
                break
    for i in range(len(interval_flux)):
        average_flux.append(np.mean(interval_flux[i]))

    interval_t=[(timestep/2)/1440.0+interval*i for i in range(tn)]

    return interval_t,average_flux

def inter_nullpt(flux,n):
    '''insert random value from others.'''
    sigma=np.nanstd(flux)
    mean=np.nanmean(flux)
    
    random_num=[]

    for i in range(len(flux)):
        if (mean-n*sigma)<=flux[i]<=(mean+n*sigma):
            random_num.append(flux[i])

    n=np.where(np.isnan(flux))
    flux=list(flux)
    for i in range(len(n[0])):
        flux[n[0][i]]=random.choice(random_num)

    return flux


def interpolation(t, flux, tfold, new_interval_num):
    '''interpolation method'''
    delta_t0 = tfold / len(t)
    delta_t = tfold / new_interval_num

    interval_it = [(delta_t / 2) + delta_t * i for i in range(new_interval_num)]

    new_iflux = [0 for i in range(new_interval_num)]

    temp = 0
    for i in range(1, len(t)):
        for j in range(temp, new_interval_num):
            if (0 < interval_it[j] < t[0]):
                new_iflux[j] = 1 + ((interval_it[j]) / (delta_t0 / 2)) * (flux[i - 1] - 1)
                continue
            elif (t[i - 1] < interval_it[j] < t[i]):
                new_iflux[j] = flux[i - 1] + ((interval_it[j] - t[i - 1]) / delta_t0) * (flux[i] - flux[i - 1])
                continue
            elif (t[-1] < interval_it[j]):
                new_iflux[j] = flux[-1] + (((interval_it[-1] - (t[-1] + delta_t0 / 2)) / delta_t0) * (1 - flux[-1]))
                continue
            elif (interval_it[j] > t[i - 1] and interval_it[j] > t[i]):
                temp = j
                break

    if np.isnan(new_iflux).sum() > 0:
        sigma = np.nanstd(new_iflux)
        mean = np.nanmean(new_iflux)
        n = 2
        random_num = []

        for i in range(len(new_iflux)):
            if (mean - n * sigma) <= new_iflux[i] <= (mean + n * sigma):
                random_num.append(new_iflux[i])

        n = np.where(np.isnan(new_iflux))
        new_iflux = list(new_iflux)
        for i in range(len(n[0])):
            new_iflux[n[0][i]] = random.choice(random_num)

    return interval_it, new_iflux

def Read_Txt():
    path =os.path.join(os.getcwd(),'kplr_data')
    dirs = os.listdir(path)
    dev_record=[]
    t_record=[]
    f_record=[]
    filename_record=[]
    count=0
    for file in dirs:
        if file[-1]=='t':
            s=time.time()
            count=count+1
            data=np.loadtxt(os.path.join(path,(str(file))),unpack=True)
            tmid,flux=delnan(data)
            tmid2,flux2 = predeal(tmid, flux, 15.0)
            t1, f1 = minus_dev(tmid2,flux2, 5)
            t_record.extend([np.asarray(t1)])
            f_record.extend([np.asarray(f1)])
            dev_record.append(np.std(f1))
            filename_record.append(np.array(file[:-4],dtype=int))
            e=time.time()
            print(count,': ',str(file),' cost time ',e-s)

    np.savetxt('dev.txt',dev_record,fmt='%10.8f')
    np.savetxt('dev_analysis_filename.txt',filename_record,fmt='%i')
    np.save('data_time_k.npy',t_record)
    np.save('data_k.npy',f_record)


def ReadTxtKm():
    path = os.getcwd() + '\Class_Kplr_star'+'\\'
    dirs = os.listdir(path)
    print(dirs)
    Km = []
    for file in dirs:
        if file[-1] == 't':
            KM = np.loadtxt(path+str(file), unpack=True)[2]
            Km.append(KM)
    np.savetxt('KM.txt', Km,fmt='%10.3f')
'''
Read_Txt()
#ReadTxtKm()
'''

for file in os.listdir(os.path.join(os.getcwd(),'kplr_data')):
    if file=='1027900.txt':

        tmid, flux, _ = np.loadtxt(os.path.join(os.path.join(os.getcwd(), 'kplr_data'), file), unpack=True)
        id = file[:-4]

        #original data plot
        plt.plot(tmid,flux,'b.',markersize=1,label=str(id))
        plt.ticklabel_format(style='sci',scilimits=(0,0), axis='y')
        plt.legend()
        plt.ylabel('flux')
        plt.xlabel('Time(day)')
        plt.title('original data')
        plt.tight_layout()
        #plt.savefig(str(id)+'-original data.png')
        plt.close()
        # 分群
        t, f = predeal(tmid, flux, 15.0)
        plt.plot(t,f,'b.',markersize=1,label=str(id))
        plt.legend()
        plt.xlabel('Time(day)')
        plt.ylabel('flux')
        plt.title('subgroup')
        plt.tight_layout()
        #plt.savefig(str(id)+'-subgroup.png')
        plt.close()

        sigma = 5
        t1, f1 = minus_dev(t, f, sigma)
        plt.plot(t1,f1,'b.',markersize=1,label=str(id))
        plt.xlabel('Time(day)')
        plt.ylabel('flux')
        plt.title('minus-'+str(sigma)+'sigma')
        plt.tight_layout()
        #plt.savefig(str(id)+'-minus-sigma.png')
        plt.close()
        from tool import ha_z
        from mangol import mandel_agol as mangol
        #signal parameters
        np.random.seed(1)
        u1 = 0.5  # linear limb darkening term
        u2 = 0  # quadratic
        khap = 3.0  # period
        rp_rs = np.random.uniform(0.06, 0.1)
        print('rprs=',rp_rs)
        a_rs = np.random.uniform(8, 35)
        iang = np.random.uniform(86, 90)
        print('iang=',iang)
        t0 = np.random.uniform(0.1, khap - 0.3)
        print('t0=',t0)
        t=np.array(t1)
        z = ha_z(t, t0, khap, a_rs, iang)
        y = mangol(z, u1, u2, rp_rs)
        sf1 = y * f1


        # folding
        p=khap*1440
        sign=['(a) p-2/1440','(b) p', '(c) p+2/1440']
        tfold = [(p - 2) / 1440, p / 1440, (p + 2) / 1440]
        fig, ax = plt.subplots(3, 1, sharex=True)
        for i in range(len(tfold)):
            '''if i!=1:
                continue'''
            t_fold1, f_fold1 = fold(tfold[i], t1, f1, 15)
            st_fold1, sf_fold1 = fold(tfold[i], t1, sf1, 15)

            if np.isnan(f_fold1).sum() == 0:
                #print(id, '-', i)
                f_fold1 = inter_nullpt(f_fold1, 2)
                plt.plot(t_fold1, f_fold1_inter, 'r.',label='inter null pt')
                plt.plot(t_fold1, f_fold1, 'b.',label='original')
                plt.legend()
                plt.xlabel('Time(day)')
                plt.ylabel('flux')
                plt.tight_layout()
                plt.savefig(str(id) + 'inter_nullpt.png')
                plt.close()

                plt.plot(t_fold1, f_fold1_inter, 'b.', label='tfold=' + str(np.around(tfold[1], 3)))
                plt.legend(loc='upper right')
                plt.xlabel('Time(day)')
                plt.ylabel('flux')
                plt.title('after folding')
                plt.tight_layout()
                plt.savefig(str(id) + 'fold.png')
                plt.show()

                # 2*1 interpolation
                fig, ax = plt.subplots(2,1, sharex=True, sharey=True)
                ax[0].plot(t_fold1, f_fold1_inter, 'r.', label='before interpolation ' + str(len(f_fold1_inter)))
                ax[0].legend(loc='upper right')
                ax[0].set_xlabel('Time(day)')
                ax[0].set_ylabel('flux')
                t_p, f_p = interpolation(t_fold1, f_fold1_inter, tfold[1], 384)
                ax[1].plot(t_p, f_p, 'b.', label='after interpolation ' + str(len(f_p)))
                ax[1].legend(loc='upper right')
                ax[1].set_xlabel('Time(day)')

                plt.tight_layout()
                fig.subplots_adjust(hspace=0,wspace=0)
                plt.savefig(str(id) + 'interpolation21..png')
                plt.close()

                # 合併interpolation

                plt.plot(t_fold1, f_fold1_inter, 'r.', label='before interpolation ' + str(len(f_fold1_inter)) + ' pt')
                plt.plot(t_p, f_p, 'b.', label='after interpolation ' + str(len(f_p)) + ' pt')
                plt.legend(loc='upper right')
                plt.title('interpolation')
                plt.xlabel('Time(day)')
                plt.ylabel('flux')
                plt.tight_layout()
                plt.savefig(str(id) + 'interpolation.png')
                plt.close()
            ax[i].plot(t_fold1,f_fold1,'steelblue',label='noise')
            ax[i].plot(st_fold1, sf_fold1, 'sandybrown', label='transit')
            ax[i].legend(loc='lower left')
            ax[i].text(0.96, 0.1, sign[i], size=10,
                       horizontalalignment='center', verticalalignment='center', transform=ax[i].transAxes)
            if i==1:
                ax[i].set_ylabel('flux')
        plt.xlabel('Time(days)')
        plt.tight_layout()
        fig.subplots_adjust(hspace=0)
        plt.savefig(str(id) + 'folding31.png')
        plt.show()

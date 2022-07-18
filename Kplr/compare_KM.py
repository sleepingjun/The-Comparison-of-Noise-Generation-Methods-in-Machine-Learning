'''
This file is to record and compare magnitude of datas we had chosen from Kepler Q1 dataset.
'''

import numpy as np
import matplotlib.pylab as plt
import os

def ReadTxtKm():
    '''collect all datas' magnitude, and save as txt file.'''
    path = os.getcwd()
    dirs = os.listdir(path)
    print(dirs)
    Km = []
    for file in dirs:
        if file[-1] == 't':
            KM = np.loadtxt(str(file), unpack=True)[2]
            Km.append(KM)
    np.savetxt('KM.txt', Km,fmt='%10.3f')

ReadTxtKm()

data=np.loadtxt('KM.txt',unpack=True)
part_data14=[data[i] for i in np.where(data<14)[0]]
part_data13=[data[i] for i in np.where(data<13)[0]]

plt.hist(data,bins='auto',alpha=0.3,cumulative=True,label='Cumulation: KM>14')
plt.hist(part_data14,bins='auto',alpha=0.3,cumulative=True,label='Cumulation: 13<KM<14')
plt.hist(part_data13,bins='auto',alpha=0.3,cumulative=True,label='Cumulation: KM<13')

plt.annotate(str(len(data)-len(part_data14)),
             xy=(15.5,8000),xytext=(13.5,8000),
             arrowprops=dict(arrowstyle='->'),
             bbox=dict(boxstyle='round,pad=0.5',fc='blue',alpha=0.2))
plt.annotate(str(len(part_data14)-len(part_data13)),
             xy=(13.5,2000),xytext=(11.5,2000),
             arrowprops=dict(arrowstyle='->'),
             bbox=dict(boxstyle='round,pad=0.5',fc='orange',alpha=0.2))
plt.annotate(str(len(part_data13)),
             xy=(12.5,650),xytext=(10.5,650),
             arrowprops=dict(arrowstyle='->'),
             bbox=dict(boxstyle='round,pad=0.5',fc='green',alpha=0.2))

plt.legend()
plt.xlabel('Magnitude')
plt.ylabel('Number')
plt.savefig('KM_cumulation_hist')
plt.show()
'''
To convert .fits to .txt
'''
import numpy as np
import os
import time
from astropy.io import fits

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

path=os.getcwd()
folder=os.path.join(path,'kplr_data')
mkdir(folder)

for file1 in dirs:
    fsize=os.path.getsize(file1)
    if (file1[-1] =='s' and fsize >0):
        hdul=fits.open(file1)
        print(hdul)
        ID=hdul[0].header['KEPLERID']
        filename=str(ID)+'.txt'
        classfilename='KR'+str(ID)+'.txt'

        K=hdul[0].header['TEFF']#Effective_temperature
        R=hdul[0].header['RADIUS']#Radius
        Km = hdul[0].header['KEPMAG']#Kepler_magnitude
        R=np.array([R],dtype=np.float32)[0]
        if 0.96<R<1.15:
            if 5200<K<6000:
                if Km<14:
                    t2 = hdul[1].data
                    JD=[]
                    flux=[]
                    dev=[]
                    for j in range(len(t2)):
                        JD.append(t2[j][0])#TTYPE1
                        flux.append(t2[j][7])
                        dev.append(t2[j][8]) #標準差
                    np.savetxt(str(folder) + '\\' + filename, np.column_stack((JD, flux, dev)), fmt='%15.8e')
                    #np.savetxt(str(newfolder1)+'\\'+filename,np.column_stack((JD,flux,dev)),fmt='%15.8e')
                    #np.savetxt(str(newfolder)+'\\'+classfilename,([K,R,Km]),fmt='%10.3f')

                else:
                    continue
            else:
                continue
        else:
            continue
        hdul.close()


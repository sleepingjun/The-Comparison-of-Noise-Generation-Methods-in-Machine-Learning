#run後會把資料夾內有的fits都轉成txt
import numpy as np
import os
import time
from astropy.io import fits


def mkdir(path):
    #判斷目錄是否存在
    #存在：True
    #不存在：False
    folder = os.path.exists(path)

    #判斷結果
    if not folder:
        #如果不存在，則建立新目錄
        os.makedirs(path)
        print('-----建立成功-----')

    else:
        #如果目錄已存在，則不建立，提示目錄已存在
        print(path+'目錄已存在')

path=os.getcwd()
'''newfolder = os.path.join(path,'Class_Kplr_star')
mkdir(newfolder)
newfolder1 = os.path.join(path,'Class_Kplr_star_txt')
mkdir(newfolder1)
dirs=os.listdir(path)'''
folder=os.path.join(path,'kplr_data')
mkdir(folder)
#count2=0
#s=time.time()
for file1 in dirs:
    fsize=os.path.getsize(file1)
    if (file1[-1] =='s' and fsize >0): #偵測fits的s，所有檔案只要是fits的就拿來做 #fsize太小的不要
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


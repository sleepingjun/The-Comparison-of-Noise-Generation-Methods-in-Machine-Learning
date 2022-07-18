import numpy as np
import matplotlib.pylab as plt
from scipy import stats
import time
a=time.time()

dev=np.loadtxt('dev.txt',unpack=True)
#file=np.loadtxt('dev_analysis_filename.txt',dtype='int',unpack=True)
#t=np.load('data_time_k.npy',allow_pickle=True)
#flux=np.load('data_k.npy',allow_pickle=True)

'''DEV=[]
index=[]
for i in range(len(dev)):
    if dev[i]<0.005:
        DEV.append(dev[i])
        index.append(i)
    else:
        continue
'''

DEV=[dev[i] for i in np.where(dev<0.005)]
n,bins,patches=plt.hist(dev,bins=4,alpha=0.7)
'''kde=stats.gaussian_kde(dev)
plt.plot(bins,kde(bins))'''
num=np.array([n],dtype=int)[0]
#plt.text(bins[1],num[0],str(num[0]),size=14)
plt.annotate(str(num[0]),
             xy=(bins[0]+bins[1]/2,2000),xytext=(bins[1]+0.0005,2000),
             arrowprops=dict(arrowstyle='<-'),
             )
plt.text((bins[1]+bins[2])/2-0.0001,num[1]+10,str(num[1]),size=11)
plt.text((bins[2]+bins[3])/2-0.0001,num[2]+10,str(num[2]),size=11)
plt.text((bins[3]+bins[4])/2-0.0001,num[3]+10,str(num[3]),size=11)
plt.title('(a)')
plt.xlabel(r'$\sigma$')
plt.ylabel('Number')
plt.ylim(0,int(max(n)))
plt.xticks(bins)
plt.savefig('dev_histgram_all')
plt.show()

'''ID=[file[i] for i in index]
tt=[t[i] for i in index]
f=[flux[i] for i in index]

np.savetxt('kepler_data_based_sigma.txt',DEV,fmt='%10.8f')
np.savetxt('kepler_data_based_filename.txt',ID,fmt='%i')
np.save('kepler_data_based_time.npy',tt)
np.save('kepler_data_based_flux.npy',f)'''

'''
#分成modelA,modelB,modelC,modelD
sigma1=[];sigma2=[];sigma3=[];sigma4=[];
ID1=[];ID2=[];ID3=[];ID4=[];
tt1=[];tt2=[];tt3=[];tt4=[];
f1=[];f2=[];f3=[];f4=[];

for i in range(len(dev)):
    if dev[i]<bins[1]:
        sigma1.append(dev[i])
        ID1.append(file[i])
        tt1.extend([t[i]])
        f1.extend([flux[i]])
    if bins[1]<=dev[i]<bins[2]:
        sigma2.append(dev[i])
        ID2.append(file[i])
        tt2.extend([t[i]])
        f2.extend([flux[i]])
    if bins[2]<=dev[i]<bins[3]:
        sigma3.append(dev[i])
        ID3.append(file[i])
        tt3.extend([t[i]])
        f3.extend([flux[i]])
    if bins[3]<=dev[i]<=bins[4]:
        sigma4.append(dev[i])
        ID4.append(file[i])
        tt4.extend([t[i]])
        f4.extend([flux[i]])

np.savetxt('modelA_dev.txt',sigma1,fmt='%10.8f')
np.savetxt('modelA_filename.txt',ID1,fmt='%i')
np.save('modelA_time.npy',tt1)
np.save('modelA_flux.npy',f1)

np.savetxt('modelB_dev.txt',sigma2,fmt='%10.8f')
np.savetxt('modelB_filename.txt',ID2,fmt='%i')
np.save('modelB_time.npy',tt2)
np.save('modelB_flux.npy',f2)

np.savetxt('modelC_dev.txt',sigma3,fmt='%10.8f')
np.savetxt('modelC_filename.txt',ID3,fmt='%i')
np.save('modelC_time.npy',tt3)
np.save('modelC_flux.npy',f3)

np.savetxt('modelD_dev.txt',sigma4,fmt='%10.8f')
np.savetxt('modelD_filename.txt',ID4,fmt='%i')
np.save('modelD_time.npy',tt4)
np.save('modelD_flux.npy',f4)


#choose max sigma of each model group.
index1=np.where(sigma1==np.max(sigma1))[0][0]
index2=np.where(sigma2==np.max(sigma2))[0][0]
index3=np.where(sigma3==np.max(sigma3))[0][0]
index4=np.where(sigma4==np.max(sigma4))[0][0]
'''
'''
index=[index1,index2,index3,index4]
sigma=[np.max(sigma1),np.max(sigma2),np.max(sigma3),np.max(sigma4)]
ID=[]
ID.append(ID1[index1])
ID.append(ID2[index2])
ID.append(ID3[index3])
ID.append(ID4[index4])

tt=[]
tt.extend([tt1[index1]])
tt.extend([tt2[index2]])
tt.extend([tt3[index3]])
tt.extend([tt4[index4]])

f=[]
f.extend([f1[index1]])
f.extend([f2[index2]])
f.extend([f3[index3]])
f.extend([f4[index4]])

np.savetxt('choose_dev.txt',sigma,fmt='%10.8f')
np.savetxt('choose_filename.txt',ID,fmt='%i')
np.save('choose_time.npy',tt)
np.save('choose_flux.npy',f)
'''

b=time.time()
print('cost ',b-a)

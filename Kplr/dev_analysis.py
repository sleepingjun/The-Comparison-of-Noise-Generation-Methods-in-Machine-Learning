'''
This file is to analyze sigma of our kepler datas.
We divided them into 4 group and hoped to have at least 50 in each group.
'''

import numpy as np
import matplotlib.pylab as plt

dev=np.loadtxt('dev.txt',unpack=True)
DEV=[dev[i] for i in np.where(dev<0.005)]

n,bins,patches=plt.hist(dev,bins=4,alpha=0.7)
num=np.array([n],dtype=int)[0]

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

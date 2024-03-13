import io
import matplotlib
matplotlib.use('Agg') 
from matplotlib import pyplot as plt
import numpy as np
import sys
temps = []
with io.open("stress", mode="r") as f:
    next(f)
    for line in f:
        temps.append(line.split())

Lx=[float(i[0]) for i in temps]
Ly=[float(i[1]) for i in temps]
Lz=[float(i[2]) for i in temps]

lam=[i[3] for i in temps]
lam=[float(i) for i in lam]


FE=[i[4] for i in temps]
FE=[float(i) for i in FE] #free energy stored in chain
deltaFE=[i[5] for i in temps]
deltaFE=[float(i) for i in deltaFE]

st0=[i[6] for i in temps]
st0=[float(i) for i in st0]
st1=[i[7] for i in temps]
st1=[float(i) for i in st1]
st2=[i[8] for i in temps]
st2=[float(i) for i in st2]
st3=[i[9] for i in temps]
st3=[float(i) for i in st3]
st4=[i[10] for i in temps]
st4=[float(i) for i in st4]
st5=[i[11] for i in temps]
st5=[float(i) for i in st5]
factor=4.11
st0=np.array(st0)*factor
st1=np.array(st1)*factor
st2=np.array(st2)*factor
st3=np.array(st3)*factor
st4=np.array(st4)*factor
st5=np.array(st5)*factor
##st6=np.array(st6)

##stop


##size=sys.getsizeof(st1)
##factor=np.zeros((1,size))
##mylist = list(xrange(10))
##factor=np.zeros((1,size))
##a=st1+st2;
##a=[0.5*i for i in a]
stress=st0-0.5*(st1+st2)
'''
plt.plot(lam,st0,label='pxx')
plt.plot(lam,st1,label='pyy')
plt.plot(lam,st2,label='pzz')
plt.plot(lam,st3,label='pxy')
plt.plot(lam,st4,label='pyz')
plt.plot(lam,st5,label='pzx')

plt.xlabel('lambda')
plt.ylabel('pressure/stress components')
plt.legend()

plt.figure()
plt.plot(lam,deltaFE,label='deltaFE')
plt.xlabel('lambda')
plt.ylabel('delta Free Energy')
plt.legend()
'''
#print(lam)
#print(deltaFE)

plt.figure()
xaxis=[(x)/Lx[0] for x in Lx]
plt.plot(xaxis,-stress,label='stress')
plt.xlabel('lambda')
plt.ylabel('Stress (MPa)')
plt.legend()

file1=open("data.txt","w")
for i in range(len(xaxis)):
    file1.write("{:7} {:7} \n".format(xaxis[i],-stress[i]))

plt.savefig("stress_lambda_x.png")

area=np.trapz(-stress, xaxis,0.001)
file2=open("area.txt","w")
file2.write(str(area))

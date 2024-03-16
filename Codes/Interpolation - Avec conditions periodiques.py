import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.interpolate import splev, splrep
plt.close("all")

def resize1Dperiodic(V, newsize):
    x1 = np.linspace(0,1,len(V), endpoint = False)
    x2 = np.linspace(0,1,newsize, endpoint = False)
    return splev(x2,splrep(x1,V, per = True, k = 1))

def resize1Dperiodic1(V, newsize):
    n = len(V)
    x2 = np.linspace(0, 1, newsize)
    xi = fftfreq(n,1/n).astype(int)[:n//2]
    ck = np.abs(fft(V)[:n//2]/(n//2))
    ck[0]*= 0.5
    pk = np.angle(fft(V)[:n//2])
    newV = np.zeros(newsize, float)
    print(pk)
    for k in range(n//2):
        newV +=  ck[k]*np.cos(2*np.pi*xi[k]*x2 + pk[k])
    return newV


 
x1 = np.linspace(0,1,40, endpoint=False)
u1 = np.sin(3*x1)
x2 = np.linspace(0,1,100, endpoint = False)
u2 = resize1Dperiodic1(u1, 100)
plt.figure(0)
plt.plot(x1,u1,"ro")
plt.plot(x2,u2)
plt.show()

x1 = np.linspace(0,1,32, endpoint=False)
u1 = np.random.random(x1.shape)
x2 = np.linspace(0,1,1000, endpoint = False)
u2 = resize1Dperiodic1(u1, 1000)
plt.figure(1)
plt.plot(x1,u1,"ro")
plt.plot(x2,u2)
plt.show()
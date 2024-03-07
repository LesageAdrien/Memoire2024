import numpy as np
import matplotlib.pyplot as plt
import scipy

plt.close("all")

def resize1Dperiodic(V, newsize):
    x1 = np.linspace(0,1,len(V))
    x2 = np.linspace(0,1,newsize)
    return splev(x2,splrep(x1,V, per = True))

x1 = np.linspace(0,1, 1000012)
u1 = np.sin(3*x1)
 
plt.figure(0)
plt.plot(x1,u1)
plt.show()

x2 = np.linspace(0,1,100)
u2 = resize1Dperiodic(u1, 100)
plt.figure(1)
plt.plot(x2,u2)
plt.show()
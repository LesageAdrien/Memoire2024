import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
plt.close("all")
v = 0.3
L = 2*np.pi
a = -L/2; b = L/2
N = 400





def u1(x):
    return (x<b-L/2).astype(float)
def u2(x):
    return 1-2*np.random.random(x.shape)
def u3(x):
    return np.exp(-300 * x**2)+ np.exp(-100 * (x+L/4)**2)
def u4(x):
    return ((x-a)/L)**40
def u5(x):
    U = x*0 +0.5
    for i in range(len(x)//2):
        U += np.sin(i*x)
    return U
def u_init(x):
    return u5(x)
def getabsfftU(U,N):
    # return np.abs(fft(np.roll(U,1)+U)/N/np.pi)[:N//2]
    return np.abs(fft(U)/N/np.pi)[:N//2]

diff_rate = 1
X, dx = np.linspace(a, b, N, endpoint = False, retstep = True)
t = 0; dt = 2/N; T = 10; i = 0

# plt.figure(3)
# plt.title("comparaison entre le facteur de diffusion du schéma discret Lax Friedrich et la diffusion continue du même facteur")
# lmbd = np.linspace(0,1,100,endpoint = False)
# plt.plot(lmbd, (1-np.cos(lmbd/2))/dt)
# plt.plot(lmbd, dx**2/2/dt*(lmbd/(2*dx))**2, "k--")
# plt.show(block = False)


U = u_init(X)
for i in range(N//2):
    newU = 0.5*( np.roll(U,1) + np.roll(U,-1) - v*dt/dx*(np.roll(U,-1)-np.roll(U,1)))
    
    if i%30 == 0:
        plt.figure(1)
        plt.clf()
        plt.plot(X,U)
        plt.show(block = False)
        plt.pause(0.01)
        
    if i ==0:
        SpU1 = getabsfftU(U,N)
        SpU2 = getabsfftU(-((1/dt) * (newU-U) + v * (np.roll(U,-1)-np.roll(U,1))/(2*dx)),N)
    else:
        SpU1 = np.vstack((SpU1,getabsfftU(U,N)))
        SpU2 = np.vstack((SpU2,getabsfftU(-((1/dt) * (newU-U) + v * (np.roll(U,-1)-np.roll(U,1))/(2*dx)),N)))
    
    U = np.copy(newU)


plt.figure(0)
plt.imshow(SpU1[::-1])
plt.colorbar()
plt.title("$|\\hat{u}(\\xi,t)|$")
plt.ylabel("t")
plt.xlabel("$\\xi$")
plt.show()
    

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
from scipy.fft import fft, fftfreq

plt.close("all")

def sproll(M,k=1):
    return sps.hstack((M[:,k:], M[:,:k]), format = "csr", dtype = float)

def Lap_mat(N,h):
    I = sps.eye(N,format = "csr", dtype = float)
    return (2*sproll(I,0) - sproll(I,1) - sproll(I,-1))/h**2


L = 2*np.pi
a = -L/2; b = L/2
N = 400

diff_rate = 1
X, dx = np.linspace(a, b, N, endpoint = False, retstep = True)

def u1(x):
    return (x<b-L/2).astype(float)
def u2(x):
    return 1-2*np.random.random(x.shape)
def u3(x):
    return np.exp(-300 * x**2)+ np.exp(-100 * (x+L/4)**2)
def u4(x):
    return ((x)/L)**40
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


t = 0; dt = 1e-14; Lt = []
theta = 0
I = sps.eye(N, dtype = float)
M = Lap_mat(N,dx)
M_im = I + dt*theta * M
M_ex = I - dt*(1-theta) * M


U = u_init(X)
SpU = getabsfftU(U,N)
for i in range(N//2):
    t += dt
    Lt.append(t)
    U = sps.linalg.spsolve(I + theta * M , M_ex.dot(U))
    SpU = np.vstack((SpU,getabsfftU(U,N)))
    



plt.figure(0)
plt.imshow(SpU[::-1])
plt.colorbar()
plt.title("$|\\hat{u}(\\xi,t)|$")
plt.ylabel("t")
plt.xlabel("$\\xi$")
plt.show(block = False)
    

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
plt.close("all")
v = 1
L = 2*np.pi
a = -L/2; b = L/2

diff_rate = 1

def u1(x):
    return (x<b-L/3).astype(float)
def u2(x):
    return 1-2*np.random.random(x.shape)
def u3(x):
    return np.exp(-300 * x**2)+ np.exp(-100 * (x+L/4)**2)
def u4(x):
    return (2*x/L)**40
def u5(x):
    U = x*0 +0.5
    for i in range(len(x)//2):
        U += np.sin(i*x)
    return U
def u_init(x):
    return u5(x)

def getFFT():
    N = 1000
    X, h = np.linspace(a, b, N, endpoint = False, retstep = True)
    U = u_init(X)
    plt.figure(0)
    plt.title("u(0,x)")
    plt.plot(X, U)
    plt.xlabel("x")
    plt.show()
    plt.pause(0.01)
    
    xfreq = fftfreq(N,1/N)[:N//2]/(b-a)*2*np.pi
    y = np.abs(fft(U)*h/np.pi)[:N//2]
    plt.figure(1)
    plt.title("$\\hat{u}(0,x)$")
    plt.plot(xfreq,y)
    plt.show(block =  False)
    
getFFT()

def dot(U,V):
    return U.dot(V)

def getAverageSymbol(N,rate):
    X, dx = np.linspace(a, b, N, endpoint = False, retstep = True)
    U = u_init(X)
    sA = 0
    s = 0
    t = 0; dt = dx**2/(2*rate); T = 2
    print("niter = "+str(T/dt))
    for i in range(100):
        t+=dt
        newU = 0.5*(np.roll(U,1)+np.roll(U,-1) - v*dt/dx*(np.roll(U,-1)-np.roll(U,1)))
        sA += dx*dot(U,-((1/dt) * (newU-U) - v * (np.roll(U,-1)-np.roll(U,1))/(2*dx)))
        s += dx*dot(U,U)
        #U = np.copy(newU)
        U = u_init(X)
    print("N= "+str(N)+" done")
    return sA/s

def getC(N_list,M_list):
    return np.hstack((M_list[0],M_list[1:]*N_list[1:]/(N_list[1:]-N_list[:-1]) - M_list[:-1]*N_list[:-1]/(N_list[1:]-N_list[:-1])))

N_list = np.linspace(50,500, 10).astype(int)
M_list = [getAverageSymbol(2*N,diff_rate) for N in N_list]
C  = getC(N_list, M_list)

plt.figure(2)
plt.plot(N_list, diff_rate*((N_list)**2))
for i in range(len(N_list)-1):
    plt.plot([N_list[i],N_list[i+1]], [C[i], C[i]])
plt.show()
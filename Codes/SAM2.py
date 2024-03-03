import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
from scipy.fft import fft, fftfreq

plt.close("all")

def sproll(M,k=1):
    return sps.hstack((M[:,k:], M[:,:k]), format = "csr", dtype = float)

def L_mat(N,h):
    I = sps.eye(N,format = "csr", dtype = float)
    return (2*sproll(I,0) - sproll(I,1) - sproll(I,-1))/h**2

"""Données relative à la discrétisation du Tore"""
a = 0; b = 2*np.pi
N = 2**10
X, h = np.linspace(a,b,N, endpoint=False, retstep = True)

xi = (fftfreq(N,1/N)/(b-a)*2*np.pi)[1:N//2]

"""Construction des matrices intervenant dans la boucle"""

def getMoy(N):
    L = L_mat(N,1/N)
    sA = 0
    s = 0
    for i in range(100):
        W = 1 - 2*np.random.random(N)
        sA = W.dot(L.dot(W))
        s = W.dot(W)
    return sA/s

#N_list = np.linspace(30,2**10,5).astype(int)
N_list = np.logspace(5,10,5, True,2).astype(int)
M_list = [getMoy(N) for N in N_list]

def getC(N_list,M_list):
    return np.hstack((M_list[0],M_list[1:]*N_list[1:]/(N_list[1:]-N_list[:-1]) - M_list[:-1]*N_list[:-1]/(N_list[1:]-N_list[:-1])))

C  = getC(N_list, M_list)

plt.figure(0)
plt.plot(xi, xi**2)
for i in range(len(N_list)-1):
    plt.plot([N_list[i],N_list[i+1]], [C[i], C[i]])
plt.show()

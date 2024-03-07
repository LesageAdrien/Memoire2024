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
N = 2**11
X, h = np.linspace(a,b,N, endpoint=False, retstep = True)
xi = fftfreq(N,1/N)[1:N//2]

"""Construction des matrices intervenant dans la boucle"""

def getMoy(N):
    L = L_mat(N,2*np.pi/N)
    sA = 0
    s = 0
    for i in range(100):
        W = 1 - 2*(np.random.random(N)<0.5)
        sA += W.dot(L.dot(W))
        s += W.dot(W)
    return sA/s
def getC(F,M):
    return np.hstack(( M[0],  ( M[1:]*F[1:] - M[:-1]*F[:-1] )/( F[1:]-F[:-1] ) ))

F = np.linspace(0,2**10,15).astype(int)[1:]
#F = np.linspace(5,10,5, True,2).astype(int)
M = [getMoy(2*freq) for freq in F]
C = getC(F, M)

plt.figure(0)
plt.plot(xi, xi**2, "k--")
plt.plot(xi, xi**2, "r--")
for i in range(len(F)-1):
    if i ==0:
        plt.plot([1,F[0]], [C[0], C[0]], "b+-")
    plt.plot([F[i]+1,F[i+1]], [C[i+1], C[i+1]], "b+-")
plt.show()

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
length = 2
a = -length*np.pi; b = length*np.pi 
N = 500
X, h = np.linspace(a,b,N, endpoint=False, retstep = True)

xi = (fftfreq(N,1/N)/(b-a)*2*np.pi)[1:N//2]

"""Construction des matrices intervenant dans la boucle"""
L = L_mat(N,h)

sA = 0
s = 0
for i in range(100):
    W = 1 - 2*np.random.random(N)
    sA = W.dot(L.dot(W))
    s = W.dot(W)
    
M1 = sA/s

plt.figure(0)
plt.plot(xi, xi**2)
plt.plot([xi[0],xi[-1]],[M1,M1])
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
from scipy.fft import fft, fftfreq
from scipy.interpolate import splev, splrep
# plt.close("all")

def sproll(M,k=1):
    return sps.hstack((M[:,k:], M[:,:k]), format = "csr", dtype = float)

def L_mat(N,h):
    I = sps.eye(N,format = "csr", dtype = float)
    return (2*sproll(I,0) - sproll(I,1) - sproll(I,-1))/h**2

"""Données relative à la discrétisation du Tore"""
# a = 0; b = 2*np.pi
# v2N = 11
# N = 2**v2N
# X, h = np.linspace(a,b,N, endpoint=False, retstep = True)


"""Construction des matrices intervenant dans la boucle"""
# def resize1D(V, newsize):
#     Vsize = len(V)
#     return interpolation.zoom(V,newsize/Vsize)

def resize1Dperiodic1(V, newsize):
    x1 = np.linspace(0, 1, len(V))
    x2 = np.linspace(0, 1, newsize)
    return splev(x2, splrep(x1, V, per=True))
def resize1Dperiodic2(V, newsize):
    n = len(V)
    x2 = np.linspace(0, 1, newsize)
    xi = fftfreq(n,1/n).astype(int)[:n//2]
    ck = np.abs(fft(V)[:n//2]/(n//2))
    ck[0]*= 0.5
    pk = np.angle(fft(V)[:n//2])
    newV = np.zeros(newsize, float)
    for k in range(n//2):
        newV +=  ck[k]*np.cos(2*np.pi*xi[k]*x2 + pk[k])
    return newV


def getMoy(N, si, dt = 1e-7, T = 1e-6):
    L = L_mat(N,2*np.pi/N)
    sA = 0
    s = 0
    for i in range(200):
        t = 0
        Winit = resize1Dperiodic2(1 - 2*(np.random.random(int(N/2**si))<0.5).astype(float), N)
        W = np.copy(Winit)
        while t < T:
            t += dt
            W -= dt * L.dot(W)
            # plt.figure(0)
            # plt.clf()
            # plt.plot(W)
            # plt.show()
            # plt.pause(0.01)
        if si == 0:
            sA += W.dot(-(W-Winit)/T)
            s += W.dot(W)
        else:
            sA += W[::si].dot(-(W-Winit)[::si]/T)
            s += W[::si].dot(W[::si])
    print("si = "+str(si)+" ... done")
    return sA/s
def getC(F,M):
    return np.hstack(( M[0],  ( M[1:]*F[1:] - M[:-1]*F[:-1] )/( F[1:]-F[:-1] ) ))

v2N = 11
N = 2**v2N
h = 2*np.pi/N
S = (np.arange(v2N))[:-2][::-1]
F = (0.5*N/(2**S)).astype(int)
M = [getMoy(N, si) for si in S]
C = getC(F, M)

xi = np.arange(N//2)
symbol_continu = xi**2
symbol_discret = 2/h**2 * (1-np.cos(h*xi))

plt.figure(1)
plt.plot(xi, symbol_continu, "k--", label = "Symbol de l'opérateur continu")
plt.plot(xi, symbol_discret, "r--", label = "Symbol de l'opérateur discret")
for i in range(len(F)-1):
    if i ==0:
        plt.plot([1,F[0]], [C[0], C[0]], "b+-")
        ms = np.mean(symbol_discret[1:F[i]])
        mc = np.mean(symbol_continu[1:F[i]])
        plt.plot([1,F[i]], [ms, ms], "ro-")
        plt.plot([1,F[i]], [mc, mc], "ko-")
        
    plt.plot([F[i]+1,F[i+1]], [C[i+1], C[i+1]], "b+-")
    ms = np.mean(symbol_discret[F[i]+1:F[i+1]])
    mc = np.mean(symbol_continu[F[i]+1:F[i+1]])
    plt.plot([F[i]+1,F[i+1]], [ms, ms], "ro-")
    plt.plot([F[i]+1,F[i+1]], [mc, mc], "ko-")
plt.legend()
plt.show()

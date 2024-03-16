import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
from scipy.fft import fft, fftfreq
from scipy.interpolate import splev, splrep
plt.close("all")

def sproll(M,k=1):
    return sps.hstack((M[:,k:], M[:,:k]), format = "csr", dtype = float)

def L_mat(N,h):
    I = sps.eye(N,format = "csr", dtype = float)
    return (2*sproll(I,0) - sproll(I,1) - sproll(I,-1))/h**2

"""Données relative à la discrétisation du Tore"""
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


def crossover1(vec1, vec2, n, N):
    result = 0
    for k in range(n):
        a = int(k/n*N)
        l = k/n*N-a
        if (a+1)<N:
            result += (1-l)*vec1[a]*vec2[a] + l*vec1[a+1]*vec2[a+1] 
        else:
            result += vec1[a]*vec2[a]
    return result
def crossover2(vec1, vec2, n, N):
    return N/n*vec1.dot(vec2)
def crossover3(vec1, vec2, n, N):
    index = np.linspace(0, N, n, endpoint = False).astype(int)
    return vec1[index].dot(vec2[index])
def crossover4(vec1, vec2, n, N):
    index = (np.linspace(0, N, n, endpoint = False)+N/n * np.random.random()).astype(int)
    return vec1[index].dot(vec2[index])
def crossover(vec1, vec2, n, N):
    return crossover1(vec1, vec2, n, N)


def getMoy(N, nf, dt = 1e-7, T = 1e-6):
    L = L_mat(N,2*np.pi/N)
    sA = 0
    s = 0
    for i in range(40):
        t = 0
        Winit = resize1Dperiodic2(1 - 2*(np.random.random(nf)<0.5).astype(float), N)
        W = np.copy(Winit)
        while t < T:
            t += dt
            W -= dt * L.dot(W)
        sA += crossover(W,-(W-Winit)/T,nf,N)
        s += crossover(W,W,nf,N)
    print("nf = "+str(nf)+" ... done")
    return sA/s
def getC(F,M):
    return np.hstack(( M[0],  ( M[1:]*F[1:] - M[:-1]*F[:-1] )/( F[1:]-F[:-1] ) ))


"""Choix des paramètres d'affichage"""
showMeans = True
showInterpotale = True
numberofmeans = 10

"""Choix des paramètres de simulation"""
N = 2**10
h = 2*np.pi/N

"""Calculs de l'approximation"""
F = np.linspace(0,N/2,numberofmeans)[1:].astype(int)
#F = np.logspace(4,10,6,  base = 2, endpoint = True).astype(int)
M = [getMoy(N, 2*nf) for nf in F]
C = getC(F, M)

"""Calcul des symboles théoriques"""
xi = np.arange(N//2)
symbol_continu = xi**2
symbol_discret = 2/h**2 * (1-np.cos(h*xi))

"""Affichage""" 
plt.figure(1)
plt.plot(xi, symbol_continu, "k--", label = "Symbol de l'opérateur continu")
plt.plot(xi, symbol_discret, "r--", label = "Symbol de l'opérateur discret")
plt.ylim(-symbol_discret[-1]*0.1,symbol_discret[-1]*1.1)
for i in range(len(F)-1):
    if i ==0:
        if showMeans:
            ms = np.mean(symbol_discret[1:F[i]])
            mc = np.mean(symbol_continu[1:F[i]])
            plt.plot([1,F[i]], [mc, mc], "ko-", label = "Moyennes locales du cas continu")
            plt.plot([1,F[i]], [ms, ms], "ro-", label = "Moyennes locales du cas discret")
        plt.plot([1,F[0]], [C[0], C[0]], "b+-", label = "Moyennes locales approximées")
    if showMeans:
        ms = np.mean(symbol_discret[F[i]+1:F[i+1]])
        mc = np.mean(symbol_continu[F[i]+1:F[i+1]])
        plt.plot([F[i]+1,F[i+1]], [mc, mc], "ko-")
        plt.plot([F[i]+1,F[i+1]], [ms, ms], "ro-")
    plt.plot([F[i]+1,F[i+1]], [C[i+1], C[i+1]], "b+-")
if showInterpotale:
    plt.plot(np.hstack((0,F[0]/2,0.5*(F[1:] + F[:-1]), F[-1])), np.hstack((C[0] - F[0] * (C[1]-C[0])/(F[2]-F[0]) ,C,C[-1] - (F[-1]-F[-2]) * (C[-1]-C[-2])/(F[-3]-F[-1]) )), "b-.")        
    plt.plot(np.hstack((F[0]/2,0.5*(F[1:] + F[:-1]))), C, "bo")
plt.legend()
plt.show()

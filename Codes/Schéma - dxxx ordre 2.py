import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
plt.close("all")

def sproll(M,k=1):
    return sps.hstack((M[:,k:], M[:,:k]), format = "csr", dtype = float)

def B_mat(N,h):
    I = sps.eye(N,format = "csr", dtype = float)
    return (-sproll(I,2) + 2*sproll(I,1) - 2*sproll(I,-1) + sproll(I,-2))/(2*h**3)



def f(x):
    return np.sin(2*x)
def fxxx(x):
    return -2*2*2*np.cos(2*x)


N = 100
a = 0; b=2*np.pi
x = np.linspace(a,b,N+1)[:-1]
h = x[1]-x[0]
U = f(x)
Uxxx = fxxx(x)
Uxxx_num = B_mat(N,h).dot(U)

plt.figure(0)
plt.plot(x,Uxxx,label = " dérivée théorique")
plt.plot(x,Uxxx_num, label = "dérivée numérique")
plt.legend()
plt.show()

def getSchemeError(f, fxxx, N):
    X = np.linspace(a,b,N+1)[:-1]
    H = X[1]-X[0]
    return np.linalg.norm(fxxx(X) - B_mat(N,H).dot(f(X)))*np.sqrt(H)


N_list = np.logspace(1.5,4, 20).astype(int)
err_list = []
for N in N_list:
    err_list.append(getSchemeError(f,fxxx,N))
    
plt.figure(1)
plt.title("Le schéma est d'ordre 2")
plt.loglog(N_list, err_list)
plt.loglog(N_list, err_list[0]*N_list[0]**2/N_list**(2))
plt.show()
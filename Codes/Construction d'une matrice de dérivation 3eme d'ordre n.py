import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
plt.close("all")

n = 4


def sproll(M,k=1):
    return sps.hstack((M[:,k:], M[:,:k]), format = "csr", dtype = float)


N = 6
a = 0; b = 1
def getB_size_N_antisym_order_n(N,n,h):
    coefs = np.linalg.solve(np.array([(np.arange(n) + 1).astype(float)**(2*i+1) for i in range(n)]), np.array([3 if i ==1 else 0 for i in range(n)]))
    I = sps.eye(N,format = "csr", dtype = float)
    B = coefs[0]*(sproll(I,-1)-sproll(I,1))/h**3
    for i in range(1,n):
        B += coefs[i]*(sproll(I,-i-1)-sproll(I,i+1))/h**3
    return B

def B_mat(N,h):
    I = sps.eye(N,format = "csr", dtype = float)
    return (-sproll(I,2) + 2*sproll(I,1) - 2*sproll(I,-1) + sproll(I,-2))/(2*h**3)


x = np.linspace(a,b,N+1)[:-1]
h = x[1]-x[0]
B = getB_size_N_antisym_order_n(N, n, h)
def f(x):
    return np.sin(2*x)
def fxxx(x):
    return -2*2*2*np.cos(2*x)


N = 100000
a = 0; b=2*np.pi
x = np.linspace(a,b,N+1)[:-1]
h = x[1]-x[0]
U = f(x)
Uxxx = fxxx(x)
Uxxx_num = getB_size_N_antisym_order_n(N, n, h).dot(U)

plt.figure(0)
plt.plot(x,Uxxx,label = " dérivée théorique")
plt.plot(x,Uxxx_num, label = "dérivée numérique")
plt.legend()
plt.show()

def getSchemeError(f, fxxx, N,n):
    X = np.linspace(a,b,N+1)[:-1]
    H = X[1]-X[0]
    return np.linalg.norm(fxxx(X) - getB_size_N_antisym_order_n(N, n, H).dot(f(X)))*np.sqrt(H)


N_list = np.logspace(1.5,4, 20).astype(int)
err_list = []
for N in N_list:
    err_list.append(getSchemeError(f,fxxx,N,n))

N_list = N_list.astype(float)
plt.figure(1)
plt.title("Le schéma est d'ordre 2")
plt.loglog(N_list, err_list)
plt.loglog(N_list, err_list[0]*N_list[0]**(2*n-2)/N_list**(2*n-2))
plt.show()
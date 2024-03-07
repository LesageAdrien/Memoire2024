import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
plt.close("all")

n = 3


def sproll(M,k=1):
    return sps.hstack((M[:,k:], M[:,:k]), format = "csr", dtype = float)


N = 6
a = 0; b = 1

def B_p(N,h,p = 1):
    I = sps.eye(N,format = "csr", dtype = float)
    return (-sproll(I,2*p) + 2*sproll(I,p) - 2*sproll(I,-p) + sproll(I,-2*p))/(2*h**3)

def getB_order_2n(N,n,h):
    coefs = np.linalg.solve(np.array([(np.arange(n) + 1).astype(float)**(2*i+1) for i in range(1,n+1)]), np.array([1 if i == 0 else 0 for i in range(n)]))
    B = coefs[0]*B_p(N, h)
    print(coefs)
    for p in range(1,n):
        B += coefs[p]*B_p(N,h,p+1)
    return B


x, h= np.linspace(a,b,N, endpoint=False, retstep = True)
B = getB_order_2n(N, n, h)
def f(x):
    return np.sin(2*x)
def fxxx(x):
    return -2*2*2*np.cos(2*x)


N = 1000
a = 0; b=2*np.pi
x, h= np.linspace(a,b,N, endpoint=False, retstep = True)
U = f(x)
Uxxx = fxxx(x)
Uxxx_num = getB_order_2n(N, n, h).dot(U)

plt.figure(0)
plt.plot(x,Uxxx,label = " dérivée théorique")
plt.plot(x,Uxxx_num, label = "dérivée numérique")
plt.legend()
plt.show()

def getSchemeError(f, fxxx, N,n):
    X, H = np.linspace(a,b,N, endpoint=False, retstep = True)
    return np.linalg.norm(fxxx(X) - getB_order_2n(N, n, H).dot(f(X)))*np.sqrt(H)


N_list = np.logspace(1.5,3, 20).astype(int)
err_list = []
for N in N_list:
    err_list.append(getSchemeError(f,fxxx,N,n))

N_list = N_list.astype(float)
plt.figure(1)
plt.title("Le schéma est d'ordre " + str(2*n))
plt.loglog(1/N_list, err_list[0]*(N_list[0]/N_list)**(2*n), "r--" , label = "$O(dx^{"+str(2*n)+"})$")
plt.loglog(1/N_list, err_list,"k+-", label = "$||err||_{L^2}$")
plt.xlabel("dx")
plt.legend()
plt.show()
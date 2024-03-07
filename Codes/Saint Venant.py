import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
def sproll(M,k=1):
    return sps.hstack((M[:,k:], M[:,:k]), format = "csr", dtype = float)
plt.close("all")

L = 10
a = -L/2; b= L/2
N = 1000
dt = 0.00001
e = 1
beta = 1

'''Définition des fonctions b,U_initial et H_initial'''
def u_init(X):
    return X*0
def zeta_init(X):
    return 0*np.exp(-4*X**2)
def b_func(X):
    return 0.4*np.exp(-2*(X-L/4)**2)

'''Définition matrices de différenciation usuelles'''
def I_mat(N):
    return sps.eye(N,format = "csr", dtype = float)
def D_mat(N,h):
    I = I_mat(N)
    return (sproll(I,-1) - sproll(I,1))/(2*h)

'''Definition des fonctions de flux discret'''
def flowU(U,H,B):
    return 0.5*e*U**2+(H+beta*B)/e
def flowH(U,H):
    return e*U*H

def fcU(U1,U2,H1,H2,B1,B2,c):
    return 0.5*(flowU(U1,H1,B1) + flowU(U2,H2,B2) - c*(U2-U1))
def fcH(U1,U2,H1,H2,c):
    return 0.5*(flowH(U1,H1) + flowH(U2,H2) - c*(H2-H1))

def dU(U,H,B,c):
        return fcU(np.roll(U,0),np.roll(U,-1),np.roll(H,0), np.roll(H,-1),np.roll(B,0),np.roll(B,-1), c) - fcU(np.roll(U,1),np.roll(U,0),np.roll(H,1), np.roll(H,0),np.roll(B,1),np.roll(B,0), c)
def dH(U,H,c):
        return fcH(np.roll(U,0),np.roll(U,-1),np.roll(H,0), np.roll(H,-1), c) - fcH(np.roll(U,1),np.roll(U,0),np.roll(H,1), np.roll(H,0), c) 
    

X , dx = np.linspace(a,b,N,endpoint = False, retstep = True)
U = u_init(X)
B = b_func(X)
H = e*zeta_init(X)-beta*B+1
#dB = 0.5/dx*(np.roll(B, -1) -  np.roll(B, 1))

ZERO = np.zeros_like(X)


rate = 1
t = 0
T = 100
c = rate * dx/dt
i = 0
while t<T:
    t+=dt

    U = U - dt/dx*dU(U,H,B,c)
    H = H - dt/dx*dH(U,H,c)

    if i%100 == 0:
        plt.figure(1)
        plt.clf()
        plt.title("t = "+str(t))
        plt.plot(X,beta*B-1, label = "fond ($\\beta b - 1$)")
        plt.plot(X,U,"r--")
        plt.plot(X,ZERO,"k--")
        plt.plot(X,H+beta*B-1, label = "surface ($\\varepsilon\\zeta$)")
        plt.legend()
        plt.show(block = i==0)
        plt.pause(0.01)

    i += 1
        
    
    
print(np.roll(np.arange(10),1))
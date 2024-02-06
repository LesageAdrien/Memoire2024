import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps

plt.close("all")

def sproll(M,k=1):
    return sps.hstack((M[:,k:], M[:,:k]), format = "csr", dtype = float)

def B_mat(N,h):
    I = sps.eye(N,format = "csr", dtype = float)
    return (-sproll(I,2) + 2*sproll(I,1) - 2*sproll(I,-1) + sproll(I,-2))/(2*h**3)

a = 0; b = 2*np.pi 
N = 1000
X, h = np.linspace(a,b,N, endpoint=False, retstep = True)

"""Construction des matrices intervenant dans la boucle"""
B = B_mat(N,h)
I = sps.eye(N,format = "csr", dtype = float)

"""Construction des données initiales et des paramètres d'évolution en temps"""
U = np.sin(2*X) + 0.1*np.sin(3*X) + 0.2 *np.sin(10 * X)
T = 10; dt = 1e-3 ; t = 0

"""Execution de la boucle"""
while t<T:
    #U = sps.linalg.spsolve(I + dt * B,  U) # Formulation implicite
    U = sps.linalg.spsolve(I + dt/2 * B,  U - dt/2 * B.dot(U))  # Formulation Crank Nichloson.
    
    plt.figure(0)
    plt.clf()
    plt.plot(X,U)
    plt.title("t = "+str(t))
    plt.show()
    plt.pause(0.01)


    


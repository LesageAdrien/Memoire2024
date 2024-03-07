import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
plt.close("all")

L = 10
a = -L/2; b = L/2
N = 1000
dt = 0.0001
e = 1
beta = 0.4

'''Définition des fonctions b,U_initial et H_initial'''
def u_init(X):
    return 0*np.exp(-3*(X+L/4)**2)+2
def zeta_init(X):
    return 2*np.exp(-50*(X+L/4)**2)
def b_func(X):
    return 0.4*np.exp(-2*(X-L/4)**2)

'''Définition matrices de différenciation usuelles'''
def I_mat(N):
    return sps.eye(N, format="csr", dtype=float)

'''Definition des fonctions de flux discret'''
def flowU(U,S):
    return 0.5*e*U**2+S/e
def flowS(U,S,B):
    return e*U*(S-beta*B+1)

def fcU(U1,U2,H1,H2,c):
    return 0.5*(flowU(U1,H1) + flowU(U2,H2) - c*(U2-U1))
def fcS(U1,U2,S1,S2,B1,B2,c):
    return 0.5*(flowS(U1,S1,B1) + flowS(U2,S2,B2) - c*(S2-S1))

def dU(U,S,c):
        return (fcU(
            np.roll(U, 0),
            np.roll(U, -1),
            np.roll(S, 0),
            np.roll(S, -1),
            c)
                - fcU(
                    np.roll(U, 1),
                    np.roll(U, 0),
                    np.roll(S, 1),
                    np.roll(S, 0),
                    c)
                )
def dS(U,S,B,c):
        return (fcS(
            np.roll(U, 0),
            np.roll(U, -1),
            np.roll(S, 0),
            np.roll(S, -1),
            np.roll(B, 0),
            np.roll(B, -1),
            c)
                - fcS(
                    np.roll(U, 1),
                    np.roll(U, 0),
                    np.roll(S, 1),
                    np.roll(S, 0),
                    np.roll(B, 1),
                    np.roll(B, 0),
                    c))

    

X, dx = np.linspace(a,b,N,endpoint = False, retstep = True)
U = u_init(X)
B = b_func(X)
S = e*zeta_init(X)
H = S-beta*B+1
#dB = 0.5/dx*(np.roll(B, -1) -  np.roll(B, 1))

ZERO = np.zeros_like(X)


rate = 1
t = 0
T = 100
c = rate * dx/dt
i = 0
while t < T:
    t += dt
    U = U - dt/dx*dU(U, S, c)
    S = S - dt/dx*dS(U, S, B, c)
    if i % 300 == 0:
        plt.figure(1)
        plt.clf()
        plt.title("t = "+str(t))
        plt.plot(X, beta*B-1, label="fond ($\\beta b - 1$)")
        plt.plot(X, U, "r--")
        plt.plot(X, ZERO, "k--")
        plt.plot(X, S, label="surface ($\\varepsilon\\zeta$)")
        plt.legend()
        plt.show(block=(i == 0))
        plt.pause(0.01)
    i += 1



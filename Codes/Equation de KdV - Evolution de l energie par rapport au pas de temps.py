import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
from scipy.fft import fft, fftfreq

plt.close("all")


def sproll(M, k=1):
    return sps.hstack((M[:, k:], M[:, :k]), format="csr", dtype=float)


def B_mat(N, h):
    I = sps.eye(N, format="csr", dtype=float)
    return (-sproll(I, 2) + 2 * sproll(I, 1) - 2 * sproll(I, -1) + sproll(I, -2)) / (2 * h ** 3)

def D_mat(N, h):
    I = sps.eye(N, format="csr", dtype=float)
    return (sproll(I, -1) - sproll(I, 1)) / (2 * h)

def L_mat(N,h):
    I = sps.eye(N,format = "csr", dtype = float)
    return (2*sproll(I,0) - sproll(I,1) - sproll(I,-1))/h**2

"""Données relative à la discrétisation  spatiale du Tore"""
length = 2
a = -length * np.pi
b = length * np.pi
N = 250
X, h = np.linspace(a, b, N, endpoint=False, retstep=True)
D = D_mat(N, h)
B = B_mat(N, h)
L = L_mat(N, h)
I = sps.eye(N, format="csr", dtype=float)
alpha = 0
theta = 0.5

def energy(U):
    return np.sum(-h*L.dot(U)*U/2 + h*U**3/6)

def u_init(x):
    return 6 * np.exp(-30 * (x) ** 2) + 6

def get_L2_growth(dt, T):
    t = 0
    U = u_init(X)
    E_init = energy(U)



    Mat = I + dt * theta * B

    """Début de boucle"""
    while t < T:
        """Calcul du prochain U"""

        # Unew = sps.linalg.spsolve(I + dt * B,  U) # Formulation implicite
        res = 1
        Uk = np.copy(U)
        i = 1
        while res > 1e-12:
            i += 1
            Uknew = sps.linalg.spsolve((alpha + 1) * Mat,
                                       alpha * Mat.dot(Uk) + U - dt * (1 - theta) * B.dot(U) - 0.5 * dt * D.dot(
                                           (Uk + U) ** 2 / 4))  # Formulation Crank Nichloson + itération de picard
            res = np.linalg.norm(Uk - Uknew) * np.sqrt(h)
            Uk = np.copy(Uknew)
        Unew = np.copy(Uk)
        """Actualisation de U et de t"""
        U = np.copy(Unew)
        t += dt
    print("i = ",i)
    return (energy(U)- E_init)


T = 1e-1
Nt_list = np.logspace(1,3.4,30).astype(int)
E_list = []
for nt in Nt_list:
    E_list.append(np.abs(get_L2_growth(T/nt,T)))
    print(nt)

plt.figure(0)
plt.title("Evolution en temps fini de J(U) en fonction du pas de temps $\\Delta_t$")
plt.loglog(T/Nt_list, E_list, "bo-", label="$J(U^n) - J(U^0)$")
plt.loglog(T/Nt_list, E_list[0]*(Nt_list[0]/Nt_list)**2, "k--", label="$\\mathcal{O}((\\Delta_t)^2)$")
plt.xlabel("$\Delta_x$")
plt.legend()
plt.show(block = True)







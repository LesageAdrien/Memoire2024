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


"""Données relative à la discrétisation du Tore"""
length = 2
a = -length * np.pi
b = length * np.pi

def soliton(c,xi):
    return 3*c/(1 + np.sinh(0.5*np.sqrt(c)*xi)**2)

def u_init(x):
    return soliton(5, x)
    #return 6 * np.exp(-30 * (x) ** 2) + 6

def get_L2_growth(N, dt, T):
    t = 0
    X, h = np.linspace(a, b, N, endpoint=False, retstep=True)
    U = u_init(X)
    L2_init = np.linalg.norm(U)*np.sqrt(h)

    """Construction des matrices intervenant dans la boucle"""
    D = D_mat(N, h)
    B = B_mat(N, h)
    I = sps.eye(N, format="csr", dtype=float)
    alpha = 0
    theta = 0.5
    Mat = I + dt * theta * B

    """Début de boucle"""
    while t < T:
        """Calcul du prochain U"""

        # Unew = sps.linalg.spsolve(I + dt * B,  U) # Formulation implicite
        res = 1
        Uk = np.copy(U)
        i = 1
        while res > 1e-8:
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
    return (np.linalg.norm(U)*np.sqrt(h))**2 - L2_init**2


N_list = np.logspace(2,3.5,10).astype(int)
L_list = []
for N in N_list:
    L_list.append(np.abs(get_L2_growth(N, 1e-3,1e-1)))
    print(N)

plt.figure(0)
plt.title("Evolution en temps fini de la norme L2 de U en fonction du pas d'espace")
plt.loglog(1/N_list, L_list, "bo-", label="$\\|U^n\\|^2_{L^2} - \\|U^0\\|^2_{L^2}$")
plt.loglog(1/N_list, L_list[0]*(N_list[0]/N_list)**2, "k--", label="$\\mathcal{O}((\\Delta_x)^2)$")
plt.loglog(1/N_list, L_list[0]*(N_list[0]/N_list)**4, "r--", label="$\\mathcal{O}((\\Delta_x)^4)$")
plt.xlabel("$\Delta_x$")
plt.legend()
plt.show()







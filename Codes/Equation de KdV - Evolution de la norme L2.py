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
a = -length * np.pi;
b = length * np.pi
N = 500
X, h = np.linspace(a, b, N, endpoint=False, retstep=True)

"""Données relative a la FFT"""

xfreq = fftfreq(N, 1 / N) / (b - a) * 2 * np.pi
firstfreq_ratio = 10
firstfreq = xfreq[1:N // firstfreq_ratio]

"""Construction des matrices intervenant dans la boucle"""
D = D_mat(N, h)
B = B_mat(N, h)
I = sps.eye(N, format="csr", dtype=float)

"""Construction des données initiales et des paramètres d'évolution en temps"""

# U = np.sin(2*X) + 0.1*np.sin(3*X) + 0.2 *np.sin(10 * X)
# U = np.sin(2 * X)
#U = 6 * np.exp(-30 * (X) ** 2) + 6
# U = (np.abs(X)< length/3).astype(float) #signal carré
# U = np.maximum(0, length/3 - np.abs(X)) #signal triangle

def soliton(c,xi):
    return 3*c/(1 + np.sinh(0.5*np.sqrt(c)*xi)**2)

U = soliton(4,X)
L2U = [np.linalg.norm(U)*np.sqrt(h)]
D3U = []
waveheight = np.max(np.abs(U))

T = 1;
dt = 1e-3;
t = 0

alpha = 0
theta = 0.5
Mat = I + dt * theta * B

dispersive_shift = (np.arctan(dt * theta * firstfreq ** 3) + np.arctan(
    dt * (1 - theta) * firstfreq ** 3)) / dt / firstfreq


"""Execution de la boucle"""
while t < T:
    """Affichage de U au temps t"""

    plt.figure(0)
    plt.clf()
    plt.plot(X, U)
    plt.ylim(-1.5 * waveheight, 1.5 * waveheight)
    plt.title("U(x,t) au temps t = " + str(round(t, 2)))
    plt.xlabel("x")
    plt.show(block = False)
    plt.pause(0.001)

    """Calcul du prochain U"""

    # Unew = sps.linalg.spsolve(I + dt * B,  U) # Formulation implicite
    res = 1
    Uk = np.copy(U)
    i = 1
    while res > 1e-6:
        i += 1
        Uknew = sps.linalg.spsolve((alpha + 1) * Mat,
                                   alpha * Mat.dot(Uk) + U - dt * (1 - theta) * B.dot(U) - 0.5 * dt * D.dot(
                                       (Uk + U) ** 2 / 4))  # Formulation Crank Nichloson + itération de picard
        res = np.linalg.norm(Uk - Uknew) * np.sqrt(h)
        Uk = np.copy(Uknew)

    print(i)
    Unew = np.copy(Uk)


    """Affichage des données concernant l'evolution de la norme L2 de U"""
    L2U.append(np.linalg.norm(Unew)*np.sqrt(h))

    L2_arr = np.array(L2U)
    plt.figure(1)
    plt.clf()
    plt.plot((L2_arr[1:]**2 - L2_arr[:-1]**2)/dt, label="$\\partial_t\\|U\\|^2_{L^2}$")
    plt.plot(D3U, "k--", label="$\\frac{(\\Delta_x)^2}{6}\\sum_{j\\in\\mathbb{Z}}(\\partial_xU)^3$")
    plt.legend()
    plt.show(block=False)
    plt.pause(0.01)

    D3U.append(h ** 2 / 6 * np.sum((((U + Unew) / 2 - np.roll((U + Unew) / 2, 1)) / h) ** 3) * h)




    """Actualisation de U et de t"""
    U = np.copy(Unew)
    t += dt





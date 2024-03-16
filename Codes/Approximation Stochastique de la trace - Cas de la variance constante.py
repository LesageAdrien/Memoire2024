import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
plt.close("all")

N=20000

"""Construction de la matrice Laplacien discret"""
def sproll(M,k=1):
    return sps.hstack((M[:,k:], M[:,:k]), format = "csr", dtype = float)
def L_mat(N,h):
    I = sps.eye(N,format = "csr", dtype = float)
    return (2*sproll(I,0) - sproll(I,1) - sproll(I,-1))/h**2
A = L_mat(N,1)

"""Choix initial des variances des coefficients"""
variances = np.ones(N, dtype = float)



"""Itérations de l'approximation de la trace"""
def getrelativeError(n_iter, variances):
    ecarts_type = np.sqrt(variances)
    sA = 0
    s = 0
    relativeErrorList = []
    for k in range(n_iter):
        v_k = (1-2*(np.random.random(N)<0.5))*ecarts_type
        sA += A.dot(v_k).dot(v_k)
        s += v_k.dot(v_k)
        relativeErrorList.append(np.abs((N*sA/s - 2*N)/(2*N)))
    return np.array(relativeErrorList)
    
n_iter1 = 50
n_iter_2 = 100

meanError = np.zeros(n_iter1, dtype = float)
for k in range(n_iter_2):
    meanError += getrelativeError(n_iter1, variances)
    print(k) if k%50 == 0 else 0
meanError*=1/n_iter_2

varError = np.zeros(n_iter1, dtype = float)
for k in range(n_iter_2):
    varError += (getrelativeError(n_iter1, variances)-meanError)**2
    print(k) if k%50 == 0 else 0
varError*=1/n_iter_2



plt.figure(0)
plt.title("Moyenne de l'erreur relative $\\varepsilon$")
plt.plot(meanError, "b+-", label = "$ \\mathbb{E}(\\varepsilon)$")
plt.plot(meanError + np.sqrt(varError), "r--", label = " $\\mathbb{E}(\\varepsilon) \\pm \\sigma$")
plt.plot(meanError - np.sqrt(varError), "r--")
plt.xlabel("itérations")
plt.legend()
plt.show()
plt.pause(0.01)
    
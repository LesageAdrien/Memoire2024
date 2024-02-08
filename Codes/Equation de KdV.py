import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
from scipy.fft import fft, fftfreq

plt.close("all")

def sproll(M,k=1):
    return sps.hstack((M[:,k:], M[:,:k]), format = "csr", dtype = float)

def B_mat(N,h):
    I = sps.eye(N,format = "csr", dtype = float)
    return (-sproll(I,2) + 2*sproll(I,1) - 2*sproll(I,-1) + sproll(I,-2))/(2*h**3)

def D_mat(N,h):
    I = sps.eye(N,format = "csr", dtype = float)
    return (sproll(I,-1) - sproll(I,1))/(2*h)


"""Données relative à la discrétisation du Tore"""
length = 2
a = -length*np.pi; b = length*np.pi 
N = 500
X, h = np.linspace(a,b,N, endpoint=False, retstep = True)


"""Données relative a la FFT"""

xfreq = fftfreq(N,1/N)/(b-a)*2*np.pi
firstfreq_ratio = 20
firstfreq = xfreq[:N//firstfreq_ratio]


"""Construction des matrices intervenant dans la boucle"""
D = D_mat(N,h)
B = B_mat(N,h)
I = sps.eye(N,format = "csr", dtype = float)


"""Construction des données initiales et des paramètres d'évolution en temps"""

#U = np.sin(2*X) + 0.1*np.sin(3*X) + 0.2 *np.sin(10 * X)
#U = np.sin(2 * X)
U = np.exp(-100*(np.sin(X/2/length))**2)
#U = (np.abs(X)< length/3).astype(float) #signal carré
#U = np.maximum(0, length/3 - np.abs(X)) #signal triangle

waveheight = np.max(np.abs(U))

T = 10; dt = 1e-3 ; t = 0

alpha = 0
theta = 0.5
Mat = I + dt*theta*B

dispersive_shift = (np.arctan(dt* theta * firstfreq**3)+np.arctan(dt*(1-theta)*firstfreq**3))/dt

"""Placement initial des fenètres"""
plt.figure(0)
plt.get_current_fig_manager().window.setGeometry(30,30,500,500)
plt.figure(1)
plt.get_current_fig_manager().window.setGeometry(30,560,500,500)
plt.figure(2) 
plt.get_current_fig_manager().window.setGeometry(560,30,870,700)
"""Execution de la boucle"""
while t<T:
    """Affichage de U au temps t"""
    
    plt.figure(0)
    plt.clf()
    plt.plot(X,U)
    plt.ylim(-1.5*waveheight,1.5*waveheight)
    plt.title("U(x,t) au temps t = "+str(round(t,2)))
    plt.xlabel("x")
    plt.show()
    plt.pause(0.001)
    
    
    
    """Calcul du prochain U"""
    
    #Unew = sps.linalg.spsolve(I + dt * B,  U) # Formulation implicite
    res = 1
    Uk = np.copy(U)
    i=1
    while res > 1e-6:
        i+=1
        Uknew = sps.linalg.spsolve((alpha + 1)*Mat, alpha*Mat.dot(Uk) + U - dt * (1-theta) * B.dot(U) - 0.5*dt*D.dot((Uk+U)**2/4))  # Formulation Crank Nichloson + itération de picard
        res = np.linalg.norm(Uk-Uknew)*np.sqrt(h)
        Uk = np.copy(Uknew)
        
    print(i)
    Unew = np.copy(Uk)


    """Affichage des données de la transformée de fourrier de la dérivée en temps."""
    
    
    y = fft(U)*h/np.pi
    
    
    plt.figure(1)
    plt.clf()
    plt.ylim(-1,1)
    plt.title("Transformée de fourier")
    plt.plot(xfreq[:N//2], np.real(y[:N//2]),"b", label= "Re(Û)") #la fonction U étant à valeurs réelles, sa FFT est une fonction paire. On ne regardera alors que la restriction à R+ de cette fonction.
    plt.plot(xfreq[:N//2], np.imag(y[:N//2]),"r", label= "Im(Û)") 
    plt.plot(xfreq[:N//2], np.abs(y[:N//2]),"k", label= "|Û|")
    plt.xlabel("$\\xi $")
    plt.legend()
    plt.show()
    plt.pause(0.001)
    
    # Le module de Û ne varie pas, mais on distingue toute de même une variation de arg(Û) au cour du temps.
    plt.figure(2) # C'est pour cela qu'on se propose ici de regarder la  dérivée de arg(Û) par rapport au temps (i.e. la vitesse de déphasage des harmoniques de U en fonction de ses fréquences)
    plt.clf()
    plt.title("Dérivée temporelle du déphasage (en rad/s) en fonction de la fréquence $\\xi$, au temps t="+str(round(t,2)) )
    plt.plot(firstfreq, np.angle(fft(Unew)[:N//firstfreq_ratio]/fft(U)[:N//firstfreq_ratio])/dt) #On ne regardera que sur les premières fréquences car sinon on risque la division par 0.
    plt.plot(firstfreq, dispersive_shift,"g--", label =  "Déphasage Dispersif du schéma")
    plt.xlabel("$\\xi $")
    plt.legend()
    plt.show()
    plt.pause(0.001)
    
    """Actualisation de U et de t"""
    U = np.copy(Unew)
    t+=dt
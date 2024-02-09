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



"""Données relative à la discrétisation du Tore"""

length = 4
a = 0; b = length * 2*np.pi 
N = 700
X, h = np.linspace(a,b,N, endpoint=False, retstep = True)



"""Données relative a la FFT"""

xfreq = fftfreq(N,h/(b-a))/(b-a)*2*np.pi
firstfreq_ratio = 20
firstfreq = xfreq[:N//firstfreq_ratio]


"""Construction des matrices intervenant dans la boucle"""

B = B_mat(N,h)
I = sps.eye(N,format = "csr", dtype = float)



"""Construction des données initiales et des paramètres d'évolution en temps"""

#U = np.sin(2*X) + 0.1*np.sin(3*X) + 0.2 *np.sin(10 * X)
#U = np.sin(10 * X)
U = 6 * np.exp(-10*(X-length*np.pi)**2)
#U = (np.abs(X)< length/3).astype(float) #signal carré
#U = np.maximum(0, length/3 - np.abs(X)) #signal triangle
T = 10; dt = 1e-2 ; t = 0
theta = 0.5  # 1 <=> Implicite, 0.5<=> Crank Nicholson
waveheight = np.max(np.abs(U))

"""Execution de la boucle"""
while t<T:
  
    
    
    """Affichage de U au temps t"""
    
    plt.figure(0)
    plt.get_current_fig_manager().window.setGeometry(30,30,500,500)
    plt.clf()
    plt.plot(X,U)
    plt.ylim(-waveheight*1.5,waveheight*1.5)
    plt.title("U(x,t) au temps t = "+str(round(t,2)))
    plt.xlabel("x")
    plt.show()
    plt.pause(0.01)
    
    
    
    """Calcul du prochain U"""
    
    #Unew = sps.linalg.spsolve(I + dt * B,  U) # Formulation implicite
    Unew = sps.linalg.spsolve(I + dt * theta * B,  U - dt* (1-theta) * B.dot(U))  # Formulation Crank Nichloson.
    
    
    """Affichage des données de la transformée de fourrier de la dérivée en temps."""
    
    
    y = fft(U)*h/np.pi
    
    
    plt.figure(1)
    plt.get_current_fig_manager().window.setGeometry(30,560,500,500)
    plt.clf()
    plt.ylim(-1,1)
    plt.title("Transformée de fourier")
    plt.plot(xfreq[:N//2], np.real(y[:N//2]),"b", label= "Re(Û)") #la fonction U étant à valeurs réelles, sa FFT est une fonction paire. On ne regardera alors que la restriction à R+ de cette fonction.
    plt.plot(xfreq[:N//2], np.imag(y[:N//2]),"r", label= "Im(Û)") 
    plt.plot(xfreq[:N//2], np.abs(y[:N//2]),"k", label= "|Û|")
    plt.xlabel("$\\xi $")
    plt.legend()
    plt.show()
    plt.pause(0.01)
    
    # Le module de Û ne varie pas, mais on distingue toute de même une variation de arg(Û) au cour du temps.
    plt.figure(2) # C'est pour cela qu'on se propose ici de regarder la  dérivée de arg(Û) par rapport au temps (i.e. la vitesse de déphasage des harmoniques de U en fonction de ses fréquences)
    plt.get_current_fig_manager().window.setGeometry(560,30,870,700)
    plt.clf()
    plt.title("Dérivée temporelle du déphasage (en rad/s) en fonction de la fréquence $\\xi$, au temps t="+str(round(t,2)) )
    
    plt.plot(firstfreq, np.arctan(dt* firstfreq**3)/dt,"k--", label =  "Déphasage des shémas Implicite et Explicite")
    plt.plot(firstfreq, (np.arctan(dt* theta * firstfreq**3)+np.arctan(dt*(1-theta)*firstfreq**3))/dt,"g-.", label =  "Déphasage du theta schéma actuel (theta = "+str(theta)+")")
    plt.plot(firstfreq, 2*np.arctan(dt/2*firstfreq**3)/dt,"r--", label =  "Dephasage du schéma de Crank Nicholson ")
    
    plt.plot(firstfreq, (firstfreq)**3, "y--", label = "Déphasage de la solution analytique")
    plt.plot(firstfreq, np.angle(fft(Unew)[:N//firstfreq_ratio]/fft(U)[:N//firstfreq_ratio])/dt, label = "Déphasage de la solution numérique") #On ne regardera que sur les premières fréquences car sinon on risque la division par 0.
    plt.xlabel("$\\xi $")
    plt.legend()
    plt.show()
    plt.pause(0.01)
    
    """Actualisation de U et de t"""
    U = np.copy(Unew)
    t+=dt

    


    


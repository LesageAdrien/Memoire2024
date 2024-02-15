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

length = 1
a = 0; b = length * 2*np.pi 
N = 700
X, h = np.linspace(a,b,N, endpoint=False, retstep = True)



"""Données relative a la FFT"""

xfreq = fftfreq(N,h/(b-a))/(b-a)*2*np.pi
firstfreq_ratio = 20
firstfreq = xfreq[1:N//firstfreq_ratio]
v_max = np.max(firstfreq**2)/2

"""Construction des matrices intervenant dans la boucle"""

B = B_mat(N,h)
I = sps.eye(N,format = "csr", dtype = float)



"""Construction des données initiales et des paramètres d'évolution en temps"""

#U = np.sin(2*X) + 0.1*np.sin(3*X) + 0.2 *np.sin(10 * X)
#U = np.sin(2 * X)
U = 6 * np.exp(-10*(X-length*np.pi)**2)
#U = (np.abs(X)< length/3).astype(float) #signal carré
#U = np.maximum(0, 3- np.abs(X - length*np.pi)) #signal triangle
T = 10; dt = 3e-4 ; t = 0

xi_for_dt2xi8_is_one = dt**(-1/4)

theta = 0.7  # 1 <=> Implicite, 0.5<=> Crank Nicholson
waveheight = np.max(np.abs(U))

"""Execution de la boucle"""
plt.figure(0)
plt.get_current_fig_manager().window.setGeometry(30,30,500,450)
plt.figure(1)
plt.get_current_fig_manager().window.setGeometry(30,560,500,500)
plt.figure(2) # C'est pour cela qu'on se propose ici de regarder la  dérivée de arg(Û) par rapport au temps (i.e. la vitesse de déphasage des harmoniques de U en fonction de ses fréquences)
plt.get_current_fig_manager().window.setGeometry(560,30,700,700)
while t<T:
  
    
    
    """Affichage de U au temps t"""
    
    plt.figure(0)
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
    plt.clf()
    plt.ylim(0,v_max)
    #plt.title("Vitesse des harmoniques en fonction de la fréquence $\\xi$, au temps t="+str(round(t,2)) )
    
    plt.plot(firstfreq, np.arctan(dt* firstfreq**3)/dt/firstfreq,"k--", label =  "Vitesses pour les shémas Implicite et Explicite ($\\theta$ = 1 et 0)")
    plt.plot(firstfreq, (np.arctan(dt* theta * firstfreq**3)+np.arctan(dt*(1-theta)*firstfreq**3))/dt/firstfreq,"g--", label =  "Vitesses pour $\\theta$ schéma actuel ($\\theta$ = "+str(theta)+")")
    plt.plot(firstfreq, 2*np.arctan(dt/2*firstfreq**3)/dt/firstfreq,"r--", label =  "Vitesses pour le schéma de Crank Nicholson ($\\theta$ = 0.5) ")
    plt.plot(firstfreq, (firstfreq)**2, "y--", label = "Vitesses pour la solution analytique")
    
    plt.plot([xi_for_dt2xi8_is_one, xi_for_dt2xi8_is_one],[0,v_max],'r', label = "$\\Delta_t^2\\xi^8 = 1$")
    plt.plot(firstfreq, np.angle(fft(Unew)[1:N//firstfreq_ratio]/fft(U)[1:N//firstfreq_ratio])/dt/firstfreq,"+", label = "Vitesses calculées numériquement") #On ne regardera que sur les premières fréquences car sinon on risque la division par 0.
    plt.xlabel("$\\xi $")
    plt.ylabel("v")
    plt.legend()
    plt.show()
    plt.pause(0.01)
    if t==0:
        break
    """Actualisation de U et de t"""
    U = np.copy(Unew)
    t+=dt

    


    


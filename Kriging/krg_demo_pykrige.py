# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 16:44:04 2025
Playing with kriging and other fancy stuff
but this time using the module pykrige instead of gstools to calculate
the kriging model
@author: abierto
"""
import numpy as np
from matplotlib import pyplot as plt
import gstools as gs
import pykrige as pkr
plt.close('all')
#esta función me permite crear campos con máximos donde me de la
#gana y perfil gausiano.
def gausianilla(x,Sig,mu,nor=1):
    """
    Devuelve el valor que toma la función de gauss para el vector
    x. No esta normalizada, así que dará un uno para x = mu

    Parameters
    ----------
    x : TYPE numpy array de shape [n,] ó [n,1] Es donde queremos
    calcular el valor de la gaussiana
        DESCRIPTION.
    Sig : TYPE Matriz definida positiva, nos permite girar y estirar
    o encoger la distribución
        DESCRIPTION.
    mu : TYPE numpy array de shape [n,] ó [n,1]
        DESCRIPTION. centro de la distribución

    Returns
    -------
    devuelve el valor de la gausiana en x
    """
    return(np.exp(-(x-mu).T@np.linalg.inv(Sig)@(x-mu))/nor)
    
def rotaynorm(theta,sigma):
    """
    Para rotar mi campo y normalizarlo. 
    """
    R =  np.array([[np.cos(theta),-np.sin(theta)],\
                   [np.sin(theta),np.cos(theta)]])
    #print('R',R)
    #print(sigma)
    sigmaR = R.T@sigma@R
    nor= np.sqrt((2*np.pi)**2*np.linalg.det(sigma))
    return(sigmaR,nor)
    
    
x = y = np.arange(-25.,26.,0.2)
#nos vamos a generar un campo y sus resultados en una malla formada
#por x e y
xm,ym = np.meshgrid(x,y)
V = np.zeros(xm.shape)
mogo = np.stack((xm,ym),axis=2)
theta = np.pi/6
sigma = np.array([[20.,0.],[0,80.]])
sigmar,norm =rotaynorm(theta,sigma)
mu = np.array([10.,10.])
 
sigma1 = np.array([[100,0],[0,100]])
theta1 = np.pi/2
sigmar1,norm1 =rotaynorm(theta1,sigma1)
mu1 = np.array([0,-18])

sigma2 = np.array([[20,0],[0,20]])
theta2 = 0
sigmar2,norm2 =rotaynorm(theta2,sigma2)
mu2 = np.array([-10,18])

for i in range(mogo.shape[0]):
    for j in range(mogo.shape[1]):
        V[i,j] = 100*gausianilla(mogo[i,j,:],sigmar,mu,norm) + \
        200*gausianilla(mogo[i,j,:],sigmar1,mu1,norm1) +\
        10*gausianilla(mogo[i,j,:],sigmar2,mu2,norm2)

#pinto en 3D mi campo
ax = plt.figure().add_subplot(projection='3d')
ax.plot_wireframe(xm,ym,V)
plt.xlabel('x')

#ahora viene lo mas importante de toda la fiesta, que es definirse
#un modelo de covarianza. En realidad, parece que define un modelo
# completo, en el que se cosidera, la covarianza, la correlación y 
#el semivariograma. La clave del modelo, esta, en el como se defina
#la correlación. Ojo a la diferencia de escala con el modelo constrido para
#gstools. Con un valor len_scal=50 este da un resultado nefasto. Creo que la
#diferencia está en que los modelos se definen distintos en uno y el otro
modelito = gs.Gaussian(dim=2,var=0.5,len_scale=5)

#generamos unos puntos al tuntun para evaluar en ellos la funcion creada
rng = np.random.default_rng(seed = 123456789)
xpyp = 50*rng.random([2,450])-25
Vp = np.zeros(xpyp.shape[1])
for i in range(xpyp.shape[1]):
    Vp[i] = 100*gausianilla(xpyp[:,i],sigmar,mu,norm) + \
    200*gausianilla(xpyp[:,i],sigmar1,mu1,norm1) +\
    10*gausianilla(xpyp[:,i],sigmar2,mu2,norm2)
#creanmos el krigenador ;)
krig = pkr.uk.UniversalKriging(xpyp[0,:],xpyp[1,:],Vp,modelito)
Vhat,var =krig.execute('grid',x,y)

#sobre la grafica de antes, pinto el campo estimado
ax.plot_wireframe(xm,ym,Vhat,color='r')
ax.scatter(xpyp[0,:],xpyp[1,:],Vp,'or')

#sacamos los resultados en plano que se ven muy bien del resultado de kriging
plt.figure()
plt.contourf(xm,ym,Vhat,30)
plt.title('campo vhat')
plt.colorbar()
plt.figure()

#pintamos juntos, curvas de nivel de ambos y puntos empleados en la regresión
plt.contour(xm,ym,Vhat,20)
plt.contour(xm,ym,V)
plt.scatter(xpyp[0,:],xpyp[1,:])

#pintamos el resutado en plano para comporar con el estimado
plt.figure()
plt.contourf(xm,ym,var,30)
plt.title('Varianza')
plt.colorbar()


plt.figure() 
plt.contourf(xm,ym,V,30)
plt.title('campo V')
plt.colorbar()

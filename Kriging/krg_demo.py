# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 16:44:04 2025
Playing with kriging and other fancy stuff
@author: abierto
"""
import numpy as np
from matplotlib import pyplot as plt
import gstools as gs
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
ax = plt.figure().add_subplot(projection='3d')
ax.plot_surface(xm,ym,V)
plt.xlabel('x')

#ahora viene lo mas importante de toda la fiesta, que es definirse
#un modelo de covarianza. En realidad, parece que define un modelo
# completo, en el que se cosidera, la covarianza, la correlación y 
#el semivariograma. La clave del modelo, esta, en el como se defina
#la correlación.
modelito = gs.Gaussian(dim=2,var=0.5,len_scale=50)

#generamos unos puntos al tuntun para evaluar en ellos la funcion creada
rng = np.random.default_rng(seed = 123456789)
xpyp = 50*rng.random([2,450])-25
Vp = np.zeros(xpyp.shape[1])
for i in range(xpyp.shape[1]):
    Vp[i] = 100*gausianilla(xpyp[:,i],sigmar,mu,norm) + \
    200*gausianilla(xpyp[:,i],sigmar1,mu1,norm1) +\
    10*gausianilla(xpyp[:,i],sigmar2,mu2,norm2)
#creanmos el krigenador ;)
krig = gs.krige.Universal(modelito,[xpyp[0,:],xpyp[1,:]],Vp,'linear')
Vhat,var =krig([x,y],mesh_type='structured')
krig.plot()
ax.plot_mesh(xm,ym,Vhat.T)

ax.scatter(xpyp[0,:],xpyp[1,:],Vp,'or')

plt.figure()
plt.contour(xm,ym,Vhat.T,20)
plt.contour(xm,ym,V)
plt.scatter(xpyp[0,:],xpyp[1,:])

plt.figure()
plt.contourf(xm,ym,var,30)


plt.figure() 
plt.contourf(xm,ym,V,30)

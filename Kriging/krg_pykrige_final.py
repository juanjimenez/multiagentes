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
from scipy.optimize import minimize as mini
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
    
def J(x,xi,xn,alpha):
    j =0
    for i in range(len(xn)):
        j += (xn[i]-x)@(xn[i]-x)
    return -alpha*j+(xi-x)@(xi-x)

def rst(x,kr,b,fmax):
    z,var = kr.execute('points',x[0],x[1])
    cos = z[0]+b*var[0]-fmax
    print(cos)
    return(cos)
    
x = y = np.arange(-25.,25.2,0.2)
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
plt.figure()
plt.contourf(xm,ym,V,30)
plt.title('campo')
plt.colorbar()
#ahora viene lo mas importante de toda la fiesta, que es definirse
#un modelo de covarianza. En realidad, parece que define un modelo
# completo, en el que se cosidera, la covarianza, la correlación y 
#el semivariograma. La clave del modelo, esta, en el como se defina
#la correlación. Ojo a la diferencia de escala con el modelo constrido para
#gstools. Con un valor len_scal=50 este da un resultado nefasto. Creo que la
#diferencia está en que los modelos se definen distintos en uno y el otro
modelito = gs.Gaussian(dim=2,var=0.5,len_scale=5)

#creanmos el krigenador ;)
datos = []
#elegimos posiciones arbitrarias de partida (cuatro agentes) Empezamos cerca
#del centro, todos comunicados siempre
pos = np.array([[0.,0.],[3.,1.],[1.,1.],[0.,10.]])
Vm = np.zeros(pos.shape[0])
for i in range(pos.shape[0]):
    Vm[i] =  100*gausianilla(pos[i],sigmar,mu,norm) + \
    200*gausianilla(pos[i],sigmar1,mu1,norm1) +\
    10*gausianilla(pos[i],sigmar2,mu2,norm2)
Vmt=Vm.copy()
#modelo inicial    
krig = pkr.uk.UniversalKriging(pos[:,0],pos[:,1],Vmt,modelito)
#Vhat,var =krig.execute('points',pos[:,0],pos[:,1])
maxvh = max(Vmt)
alpha = 0.01
b = 0.2

bnds =([-25,25],[-25,25])

for i in range(20):
    par =(krig,b,maxvh)
    const = {'type': 'ineq', 'fun': rst, 'args': par }
    res = np.zeros([4,2])
    for d in range(pos[-4:].shape[0]):
        r = mini(J,pos[d],(pos[d],np.delete(pos,d,0),alpha),bounds=bnds,constraints=const)
        res[d] = r.x
        vx,varx =krig.execute('points',r.x[0],r.x[1])
        print('ext',vx[0]+b*varx[0]-maxvh)
#sacamos los resultados en plano que se ven muy bien del resultado de kriging
    if i%2:
        Vhatms,varms = krig.execute('grid',x,y)
        plt.figure()
        plt.contourf(xm,ym,Vhatms,30)
        plt.title('campo vhat')
        plt.scatter(pos[:,0],pos[:,1],color='w')
        plt.scatter(res[:,0],res[:,1],color='r')
        plt.colorbar()

    pos = np.append(pos,res,0)
# plt.figure()
    for i in range(res.shape[0]):
        Vm[i] =  100*gausianilla(res[i],sigmar,mu,norm) + \
            200*gausianilla(res[i],sigmar1,mu1,norm1) +\
            10*gausianilla(res[i],sigmar2,mu2,norm2)
    Vmt= np.append(Vmt,Vm)    
    krig = pkr.uk.UniversalKriging(pos[:,0],pos[:,1],Vmt,modelito)
    maxvh = max(Vmt)
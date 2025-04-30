# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 12:53:12 2025
Kriging utilities
@author: abierto
"""

import numpy as np
from matplotlib import pyplot as plt
import gstools as gs
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
        DESCRIPTION.plt.title('campo V')
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
    """
    Funcion  de coste para seleccionar un nuevo punto de medida
    Parameters
    ----------
    x : TYPE doble 
        DESCRIPTION. posición nueva 
    xi : TYPE doble
        DESCRIPTION. Posición actual
    xn : TYPE double
        DESCRIPTION. Array con las posiciones de los vecinos
    alpha : TYPE doble
        DESCRIPTION. peso que se da a la suma de distancias de los vecinos al
        nuevo punto de medida

    Returns
    -------
    TYPE double
        DESCRIPTION. valor de la funcion de coste en x
    
    """
    j =0
    for i in range(len(xn)):
        j += (xn[i]-x)@(xn[i]-x)
    return -alpha*j+(xi-x)@(xi-x)

def rst(x,kr,b,fmax,method):
    """
    Restriccion

    Parameters
    ----------
    x : TYPE double
        DESCRIPTION. posición para nueva medida
    kr : TYPE función
        DESCRIPTION. función para cálculo de estimacion del campo por kriging
    b : TYPE double
        DESCRIPTION. Peso que se da a la varianza
    fmax : TYPE double
        DESCRIPTION. Valor máximo del campo medido hasta el momento

    Returns
    -------
    valor de la restriccion en el punto x debería ser siempre > 0

    """
    z,var = kr([x[0],x[1]])
    cos = z[0]+b*var[0]-fmax
    #print(cos)
    return cos

def evalJ(x,y,xi,xn,alpha):
    xm = x.shape[0]
    ym= y.shape[0]
    Jv = np.zeros([xm,ym])
    for i in range(xm):
        for j in range(ym):
            p = np.array([x[i],y[j]])
            Jv[i,j] = J(p,xi,xn,alpha)
            print(p@p)
            print(Jv[i,j])
    return Jv

def Jrst(x,xi,xn,alpha,kr,b,fmax):
    j =0
    for i in range(len(xn)):
        j += (xn[i]-x)@(xn[i]-x)
    z,var = kr([x[0],x[1]])
    cos = z[0]+b*var[0]**0.5-fmax    
    sal = -alpha*j+(xi-x)@(xi-x) + 1e6*(cos<=0)
    #print(sal)
    return sal
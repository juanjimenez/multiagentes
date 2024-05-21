# -*- coding: utf-8 -*-
"""
Created on Sat May 18 19:48:21 2024
Ejmplillo de sistema multiagente que converge a una treayectoria.
La dinamica de los agentes corresponde a u uniciclo
@author: abierto
"""

import numpy as np
from scipy.integrate import solve_ivp as sl
import matplotlib.pyplot as plt


###################funciones para definir la trayectoria######################
def fun(w):
    '''
    Esta función debe definir una curva en 3D que seguirán los agentes
    debe estar definida en funcion de la coordenada w (augmented field)
    '''
    #f1 = 15*np.sin(2*w)
    #f2 = 30*np.sin(w)*(np.sqrt(0.5*(1-0.5*(np.sin(w))**2)))
    #f3 = 3+5*np.cos(2*w)
    f1 = 10*np.sin(w) + 20*np.sin(2*w)
    f2 = 10*np.cos(w)-20*np.cos(2*w)
    f3 = -20*np.sin(3*w)
    return np.array([[f1], [f2], [f3]])

def dfun(w):
    '''
    Derivada primera de la función anterior, La defino explícitamente
    '''
    df1 = 10*np.cos(w) + 40*np.cos(2*w)
    df2 = -10*np.sin(w) + 40*np.sin(2*w)
    df3 = -60*np.cos(3*w)
    return np.array([[df1], [df2], [df3]])
    
def ddfun(w):
    '''
    Derivada segunda de la función fun, La defino explícitamente
    '''
    ddf1 = -10*np.sin(w) - 40*np.sin(2*w)
    ddf2 = -10*np.cos(w) + 80*np.cos(2*w)
    ddf3 = 180*np.sin(3*w)
    return np.array([[ddf1], [ddf2], [ddf3]])

#matriz lalpaciana de un grafo conexion en daisy chain
def grafo_L(N):
    '''
    Cada agente solo se relaciona con el anterior y el siguiente, formando 
    una cadena cerrada

    '''
    L =2*np.eye(N)
    sd = -np.ones([N-1])
    L = L + np.diag(sd,1)+np.diag(sd,-1)
    
    # Llenar las esquinas con -1
    L[0, -1] = -1
    L[-1, 0] = -1
    return L


def din_agen(N,x,K,kc,Kth,L,v,E,wstr):
    ''' 
    Esta funcion define la dinamica de un agente
    incluye la guia del campo y el efecto de la  cordinación
    Asumo que la dimensión de movimiento en R^3, por tanto (-1)^n = (-1)^3 = -1
    '''
    
    c = -L@(x[:,4] - wstr)
    
    for i in range(N):
        #calculamos el valor del campo guía en la posición del agente
        xi =x[i:i+1,:].T
        w = xi[3,0]
        p = xi[0:3]
        f = fun(w)
        df = dfun(w)
        ddf = ddfun(w)
        Kphi = K@(p-f)
        Kdf  = K@df
        Xi = np.vstack((-df-Kphi,-1+Kphi.T@df + kc*c[i]))
        nXi = np.linalg.norm(Xi)
        Xib = Xi/nXi
        nXip = np.linalg.norm(Xi[0:2])
        bXip = xi[0:2]/nXip
        Jpf =np.hstack( \
            (np.vstack((-K,Kdf.T)),\
             np.zeros((4,i)),\
             np.vstack((-ddf + K@df,Kphi.T@ddf-np.ones((1,3))@K@df**2)),np.zeros((4,N-1-i))))
        Jcr = np.hstack((np.zeros((4,3)),np.vstack((np.zeros((3,N)),-L[i,:]))))
        Jxip = (np.eye(4)-Xib@Xib.T)@(Jpf+kc*Jcr)/Xib
        
    
    
    
    
        #definimos la dinámica y la ley de control
        dp1 = v*np.cos(xi[5]) #x
        dp2 = v*np.sin(xi[5]) #y
        dp3 = v*xi[2]/nXip    #z
        dp4 = v*xi[4]/nXip    #w (la coordenada virtual)
        dp5 = -(Xip/nXip).T@E@Xipd/nXi #theta       
    
    
###########################Integracion del modelo#############################


#Parámetro de simulación 
#dimensiones del espacio 3D
N = 20           #nº de agentes 
K =np.diag([1,1,2]) #ganancias
kc = 200 
L = grafo_L(N) #laplaciana daisy chain de la formacion
v = 2 #velocidad fija de los uniciclos
#coordenadas iniciales de los robots 
#pos = np.random.rand(N, n)*100 #filas: dimensiones
                           #columnas: nº de robots
p = np.random.randint(-50, 51, size=(N, 3))
theta = 2*np.pi*np.random.random((N,1))

#añadimos a la matriz de posiciones la coordenada virtual w 
w = np.ones((N,1)) #ejemplo: todos valen 1
wstr = np.arange(0,2*np.pi,2*np.pi/N)    #distribución deseada
x = np.hstack((p,w,theta))

sol = sl(vector_field, (0,200),Xi[0],method='LSODA',args=(ki,n,N,ww,kc,L),max_step=0.2) 

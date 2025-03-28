#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:23:55 2025
Playing with kriging 
@author: juan
"""
import numpy as np
import numpy.linalg as lina
import matplotlib.pyplot as pl

#llamo semivariograma a la función de correlación que me da tito Kahn en el
#artículo. 
def semivar(x1,x2,sigmaz=0.5,thetai=[50,50],qi=[2,2]):
    """
    Este semivariograma ¿? es directamente la función de correlacion de tito Kahn
    en el artículo. puedo dar un valor distinto de thetai a cada coordenada
    y tambien un valor distinto al qi . Esto me da un cierto control sobre
    la (posible) anisotropía de la correlación pero a ver quien es el guapo que 
    modela con esos dos parametros.
    Calcula cual sería el valor de la covarianza entre dos puntos x1 y x2
    ojo que el parámetro que usa es sigmaz no sigmaz al cuadrado
    """
    ji = sigmaz**2*np.exp(-sum(np.abs((x1-x2))**qi/thetai))
    #print(ji)
    return ji

def parametros(xm,ym,semiv,fun = None):
    """
    Esta función calcula parametros necesarios para la regresión a partir
    de una tabla de medidas (xm==puntos, ym valor del campo en los puntos)

    Parameters
    ----------
    xm : TYPE Matriz nX2 con los puntos en los que se ha medido
        DESCRIPTION.
    ym : TYPE vector nX1 con las medidas en dichos puntos
        DESCRIPTION.
    semiv : TYPE la funcion para calcular el semivariograma. Se puede usar 
    semivar y de hecho es la que uso. Peero es una gran guarrada porque como
    no pasamos sus parámetros solo se pueden usar los paramentros por defecto
    xP
        DESCRIPTION.
    fun : TYPE, optional. Funcion de kernel si se va a usar krigin universal
        DESCRIPTION. The default is None.

    Returns
    -------
    list
        DESCRIPTION. beta, el un array con los coeficientes beta del regresor
        calculado
        Kinv es el inverso de la matriz de covarianza K, coef es el producto
        K^(-1)(y-R*beta) que basta con calcularlo una vez ya que siempre el
        igual al calcular la regresión

    """
    if fun == None:
        R = xm       
    else:
        #supongo que me defines la funcion de modo que devuelve R directamente
        #esto no lo he usado nunca...
        R =fun(xm)
    #contamos los puntos medidos     
    numdat =xm.shape[0]
    #me creo una matriz de covarianza de zeros pa' ir rellenado
    K = np.zeros((numdat,numdat))
    #relleno la matriz
    for i in range(numdat):
        for j in range(i,numdat):
            K[i,j] = semivar(xm[i],xm[j])
            K[j,i] = K[i,j]
            #ojo supongo que no ahy error en la medida sigmaw=0
            #aunque tampoco sería difícil de añadir...
    Kinv = lina.inv(K)
    #la obtención de beta está en las notas que os mandé
    beta = lina.inv(R.T@Kinv@R)@R.T@Kinv@ym
    coef = Kinv@(ym-R@beta)
    return [beta,Kinv,coef]

def regresor(x,xm,beta,Kinv,coef,semivar,sigmaz=0.5):
    """
    Con todos los ingredientes calcular ahora la regresión es m&s ;)
    

    Parameters
    ----------
    x : TYPE Punto en que se quiere estimar el valor del campo se puede pasar
    una matriz nX2 de puntos
        DESCRIPTION.
    xm : TYPE Puntos en los que tenemos medidas
        DESCRIPTION. matriz nX2
    beta : TYPE Calculado con la función parámetro
        DESCRIPTION.
    Kinv : TYPE idem al anterior
        DESCRIPTION.
    coef : TYPE iden al anterior
        DESCRIPTION.  
    semivar : TYPE Funcion del semivariograma, con el mismo problema de 
    antes, no se le pueden definir los parámetros solo usa los valores por
    defecto
        DESCRIPTION.
    sigmaz : TYPE, optional de nuevo, es la sigmaz, no su cuadrado
        DESCRIPTION. The default is 0.5.

    Returns
    -------
    phihat : TYPE valor de la funcion en x (estimada)
        DESCRIPTION.
    sigphi2 : TYPE valor de la varianza en x (tb estimada)
        DESCRIPTION.

    """
    #contamos puntos medidos
    numdat = xm.shape[0]
    rdat = range(numdat)
    #contamos puntos en que se quiere obtener la estima
    numreg =x.shape[0]
    rreg = range(numreg)
    #vector de ceros para guardar als convarianzas
    kx = np.zeros(numdat)
    #vector de ceros para guardar las salidas
    phihat = np.zeros(numreg)
    #vector de ceros para guardar las variazas en los puntos calculados
    sigphi2 = np.zeros(numreg)
    for i in rreg:
        #calculamos las Kx
        for j in rdat:
            kx[j] = semivar(x[i],xm[j])
        #calcualamos los valores de la regresión y la variaza
        phihat[i] = x[i]@beta + kx@coef
        sigphi2[i] = sigmaz**2*(1-kx@Kinv@kx)
    return phihat ,sigphi2   
        
def dibujines(x,y,xm,ym):
    #nunca la he usado, la verdad.
    fig = pl.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x[:,0],x[:,1],y,marker='^')
    ax.scatter(xm[:,0],xm[:,1],ym,marker='o')
       

     
    


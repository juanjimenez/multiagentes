# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 20:21:15 2022

@author: abierto
Movida de calcular los pesos para conseguir convergencia a una transformacion
Afin etc,etc...
"""
import numpy as np
import picos as pic

def M_ibuilder(pstar,B):
    #pstar es el vector de configuracion de referencia para la formacion
    #B es la matriz de incidencia. Hay que revisar pero creo que Shiyu define
    #la matriz de incidencia como la traspuesta de la nuestra.
    #hago los calculos para la nuestra OJO OJO
    
    np.dpt(np.dot(pstar.T,diag(B[i]),
    
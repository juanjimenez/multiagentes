# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 16:44:04 2025
Playing with kriging and other fancy stuff
Calculado con los parametros del artículo para ver qué sale
kriginng_util tiene una función para generar gausianas, centro y varianza a 
definir y también con la posibilidad de rotarlas
 las funciones auxiliares para definir J (la funcion
que s equiere optimizar y rest (la restriccion). Ver el artículo de Athur Kanh
hal-01170131v2)
 
En este caso se ha optimizado empleando direct. Este metodo busca el mínimo
global en un entorno acotado (límites a las coordenadas del espacio de busqueda)
Para unir la función a optimizar con las restricciones se ha creado una única
función Jrst. El truco es que si no se cumple la restricción, a la función de
coste se le añade un valor 10^6. Esto hace que el mínimo caiga fuera de los
límites que se exploran. por tanto direct desechará la regiones donde pase esto
y solo buscará donde la restricción se cumpla ( a al menos, eso creo)
@author: abierto
"""
from kriging_util import *
from scipy.optimize import direct
#nos vamos a generar un campo y sus resultados en una malla formada
#por x e y
x = y = np.arange(-25.,25.2,0.2)
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
        V[i,j] = 1000*gausianilla(mogo[i,j,:],sigmar,mu,norm) + \
        2000*gausianilla(mogo[i,j,:],sigmar1,mu1,norm1) +\
        100*gausianilla(mogo[i,j,:],sigmar2,mu2,norm2)
ax = plt.figure().add_subplot(projection='3d')
((ax.plot_wireframe(xm,ym,V)))
plt.xlabel('x')
plt.figure() 
c = plt.contourf(xm,ym,V,30) #campo real
plt.title('campo V') 


#ahora viene lo mas importante de toda la fiesta, que es definirse
#un modelo de covarianza. En realidad, parece que define un modelo
# completo, en el que se considera, la covarianza, la correlación y 
#el semivariograma. La clave del modelo, esta, en el como se defina
#la correlación.
modelito = gs.Gaussian(dim=2,var=0.5,len_scale=50)
rng = np.random.default_rng(52345) #semilla para generar inicios aleatorios reproducibles
for u in range(10):
    #generamos unos puntos al tuntun para evaluar en ellos la funcion creada
    #pos = np.array([[-10.,20.],[20.,20.],[20.,-20.],[-20.,-20.]])
    #pos = np.array([[-10.,8.],[1.,1.],[1.,-1.],[-1.,-1.]])
    #pos = np.array([[-1.,1.],[1.,1.],[10.,-4.],[-1.,-1.]]) #Esta va muy bien
    
    pos = rng.random((4,2))*50.-25.
    #mido el campo en los puntos que tengo
    Vm = np.zeros(pos.shape[0])
    for i in range(pos.shape[0]):
        Vm[i] =  1000*gausianilla(pos[i],sigmar,mu,norm) + \
        2000*gausianilla(pos[i],sigmar1,mu1,norm1) +\
        100*gausianilla(pos[i],sigmar2,mu2,norm2)
    #Vmyt va a contener todas la medidas que se realizen a lo largo de la búqueda
    Vmt=Vm.copy()
    
    #creanmos el krigenador con los puntos medidos ;)
    krig = gs.krige.Universal(modelito,[pos[:,0],pos[:,1]],Vmt,'linear')
    
    #y optenemos estima de campo y varianza sobro todos los puntos de nuestra 
    #cuadrícula de trabajo 
    Vhat,var =krig([x,y],mesh_type='structured')
    
    #krig.plot() #le dejo pintar a el el campo estimado
    #ax.plot_wireframe(xm,ym,Vhat.T,color='r')
    plt.figure()
    #por como creo los valores del campo con bucles en x e y, la matriz
    #que optengo es la traspuesta a la real XP. Todo:arreglar la chapuza.
    #lo pinto para comparar con el que pinta krig y ver que lo que digo es cierto
    plt.subplot(1,2,1)
    c = plt.contourf(xm,ym,Vhat.T,90)
    plt.title('campo vhat')
    
    #pinto sobre el campo los puntos iniciales
    plt.scatter(pos[:,0],pos[:,1],color='r')
    plt.colorbar(c)
    
    #calculo el valor máximo medido so far y la posición
    maxvh = max(Vmt)
    pmax = pos[np.where(Vmt.flat == max(Vmt))] [0]
    
    #####Parámetros del modelo de búsqueda#########################################
    alpha = 0.1 #0.25 #peso de la suma de la distancia del punto a los vecinos 
    b = 200 #peso que se da a la varianza en la función de restricción
    ###############################################################################
    
    ####Parametros del modelo dinámico#############################################
    M = 1 
    v0 = np.array([[0,0,0,0],[0,0,0,0]])
    k1 = 47
    k2 = 5 #50 original
    k3 = 1600
    q = 0.1
    tk = 0.01
    ###############################################################################
    #valor de la restricción sobre todos los puntos de la cuadrícula
    cs = Vhat+b*var**0.5-maxvh
    
    bnds =((-25,25),(-25,25)) #cotas en x e y del D, el espacio de búsqueda
    
    #calculamos un primer destino
    ############optimizador####################################################
    # maxfun está de hecho en su valor por defecto = 1000n (n dimension del es
    #pacio de busqueda) (numero maximo de evaluaciones de la funcion objetivo)
    #la len_tol define el tamño de cuadricula (lado) por debajo del cual se considera
    #que el algoritmo ha convergido por defecto es 1e-6. Si se baja el valor
    # en algunos pasos no converge porque agota maxfun
    #ver ayuda de scipy. es posible que se pueda ajustar mejor
    #muy importante siempre se calcula la nueva posición para el cuarto
    #agente empezando a contar or la cola ret[0]. De este modo, cada cuatro
    #iteraciones se han renovado las posiciones de los cuatro agentes.
    visitados = pos.copy()
    #separamos puntos visitados por agentes
    vagen = []
    poscum = []
    for i in pos:
        vagen.append(np.array([i]))
        poscum.append(i)    
            
    
    
    destinos = pos.copy()
    for i in range(destinos.shape[0]):
        r = direct(Jrst,bnds,args =(pos[i],np.delete(pos,i,axis=0),alpha,krig,b,maxvh),maxfun = 2000,len_tol=1e-3)
        destinos[i] = r.x
    
       
       
    
    it = 0 #iniciamos unn contador para saber cuantas iteraciones está haciendo
    
    ###############Inicio Bucle de Búsqueda########################################
    while (max(cs.flat) > 0)&(it < 100):
        #vamos a calcular mientras se cumpla la condición en los puntos de la
        #cuadricula, de que al menos hay uno en que se cumple la restricción. 
        #Se podría apretar más pero creo que no tiene sentido
        #El número de iteraciones máximo es para proteger de bucles infinitos
        #Si se ha evaluado el campo en 100 puntos el algorithmo es muuuy malo
        
        #los movemos hasta que uno se acerca a sus destino
        while all(np.sqrt(np.sum((destinos-pos)*(destinos-pos),axis=1))>0.01):
            x0,v0 = dinamica(M,v0,pos.T,destinos.T,k1,k2,k3,q,tk)
            pos = x0.T
        for i in range(len(poscum)):
            poscum[i] = np.vstack((poscum[i],pos[i]))
        #vemos quien ha llegado a destino
        windex= np.nonzero(np.sqrt(np.sum((destinos-pos)*(destinos-pos),axis=1))<=0.01)[0][0]
        #medimos en el punto nuevo 
        Vm =  1000*gausianilla(pos[windex],sigmar,mu,norm) + \
                2000*gausianilla(pos[windex],sigmar1,mu1,norm1) +\
                100*gausianilla(pos[windex],sigmar2,mu2,norm2)
        
        #si la nueva medida es el valor mas grande obtenido so far, lo guardamos
        #Así como la posicion        
        if Vm > maxvh:
            maxvh = Vm
            pmax = pos[windex] #aqui se guarda la posición del máximo
        #Añadimos la medida a las que ya tenemos
        Vmt= np.append(Vmt,Vm)
        
        #añadimos el punto a la lista de visitados        
        visitados = np.vstack((visitados,pos[windex]))
        vagen[windex] = np.vstack((vagen[windex],pos[windex]))
        #y calculamos un nuevo modelo para kriging, añadiendo la nueva posicion
        #y la nueva medida    
        
        
        krig = gs.krige.Universal(modelito,[visitados[:,0],visitados[:,1]],Vmt,'linear')
        
        
        for i in range(destinos.shape[0]):
            r = direct(Jrst,bnds,args =(pos[i],np.delete(destinos,i,axis=0),alpha,krig,b,maxvh),maxfun = 2000,len_tol=1e-3)
            destinos[i] = r.x
        ###########################################################################
        
        #usamos el modelo para volver a estimar en la cuadricula
        Vhatms,varms = krig([x,y],mesh_type='structured')
        #y así obtener una nueva condición de parada, que al menos haya un punto 
        # de la restricción por encima de cero
        for i in range(Vhatms.shape[0]):
            for j in range(Vhatms.shape[1]):
                if Vhatms[i,j] < 0:
                    Vhatms[i,j] = 0 
        cs = Vhatms+b*varms**0.5-maxvh
        
        #evaluamos la estima de campo y varianza en el punto encontrado
        vx,varx =krig([pos[windex,0],pos[windex,1]])
        #imprimimos info de lo que está pasando. Se puede omitir para que 
        #calcule más deprisa
        #print(f'it={it}\nmax={max(cs.flat)}\nrest={vx[0]+b*varx[0]-maxvh}\n{r}')
        print(it)   
        #añadimos el punto a las posiciones visitadas
            
            
        
        
        
        
        
        
        #de vez en cuando (ahora cada cinco iteraciones) dibujamos los resultados
        #permite ver lo que pasa con los campos estimados pero consume recursos...
        #y tiempo. posiblemente lo menor es comentarlo, o diezmar las figuras
        #obtenidas si necesita muchas iteraciones
        # if i%10==0:
            
        #     plt.figure()
        #     c = plt.contourf(xm,ym,Vhatms.T,30)
        #     plt.title('campo vhat')
        #     plt.scatter(pos[:,0],pos[:,1],color='w')
        #     plt.scatter(pos[-1,0],pos[-1,1],color='r')
        #     plt.colorbar(c)
            
        #incrementamos el contador
        it += 1
    ############## Fin del bucle de busqueda ######################################
    
    ###############forensic report#################################################
    
    #pintamos el campo real
    colorin = ['c','r','k','w','b','m','g','y']
    
    
    
    
    
    #este calculo sobra es el mimo que el de la linea 156
    Vhatms,varms = krig([x,y],mesh_type='structured')
    #Eliminamos la parte negatica del campo estimado (la hacemos cero). La idea es resaltar la parte
    #de la estima que tiene sentido, nuestro campo gaussiano no tiene valores
    #negativos. Si los tuviera esto no debería hacerse ¿?
    Vhatfinal = Vhatms.copy()
    for i in range(Vhatfinal.shape[0]):
        for j in range(Vhatfinal.shape[1]):
            if Vhatfinal[i,j] < 0:
                Vhatfinal[i,j] = 0 
    
    #pintamos la varianza final
    csfinal = cs.copy()
    for i in range(csfinal.shape[0]):
        for j in range(csfinal.shape[1]):
            if csfinal[i,j] < 0:
                csfinal[i,j] = 0 
    
    plt.subplot(1,2,2)
    c=plt.contourf(xm,ym,csfinal.T,30)
    plt.title('varianza')
    plt.colorbar(c)
    
    plt.figure()
    plt.subplot(1,2,1)
    c = plt.contourf(xm,ym,V,30) #campo real
    plt.title('campo V') 
    for i in  range(len(vagen)):
          plt.scatter(vagen[i][:,0],vagen[i][:,1],color = colorin[i]) #puntos recorridos por todos los agentes
          plt.plot(poscum[i][:,0],poscum[i][:,1],color = 'w')
          plt.scatter(pmax[0],pmax[1],color='m',marker ='*',s = 10) #maximo encontrado
    plt.colorbar(c)
    
    #pintamos el campo estimado
    
    plt.subplot(1,2,2)
    c =plt.contourf(xm,ym,Vhatfinal.T,30) #campo estimado
    plt.title('campo vhat it ='+str(it))
    for i in  range(len(vagen)):
        plt.scatter(vagen[i][:,0],vagen[i][:,1],color = colorin[i]) #puntos recorridos por todos los agentes
        plt.plot(poscum[i][:,0],poscum[i][:,1],color = 'w')
    plt.scatter(pmax[0],pmax[1],color='m',marker ='*') #posición del máximo encontrado
    plt.colorbar(c)
    #Iteración en que se encontro el maximo (ojo en realidad es medida en que se
    #encontró)
    
    print(np.where(Vmt == maxvh))



    
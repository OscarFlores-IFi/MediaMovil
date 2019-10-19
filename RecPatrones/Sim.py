# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 17:51:08 2019

@author: Oscar Flores
"""
#%% Importar Librerías. 
import pickle 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Simulacion import Optimizacion
from Simulacion import Graficos
from Simulacion import Genetico
from Simulacion import Kclusters


simulacion = Optimizacion.simulacion
grafico = Graficos.simulacion
genetico = Genetico.genetico
k_clusters = Kclusters.k_clusters
#%%############################################################################
############################  Crear model_close ###############################'ALFAA.MX',
############################################################################### 

#%% Datos en csv
csv = ['AMXL.MX','WALMEX.MX','TLEVISACPO.MX','GMEXICOB.MX','GFNORTEO.MX','CEMEXCPO.MX','PENOLES.MX','GFINBURO.MX','ELEKTRA.MX','BIMBOA.MX','AC.MX','KIMBERA.MX','LABB.MX','LIVEPOL1.MX','ASURB.MX','GAPB.MX','ALPEKA.MX','GRUMAB.MX','ALSEA.MX','GCARSOA1.MX','PINFRA.MX']
for i in np.arange(len(csv)):
    csv[i] = '../Data/%s.csv'%csv[i]

#ndias = [3,5,8,13,21,34,55,89,144]
ndias = [5,20,40,125]
n_clusters = 4
#%% Guardamos model_close en model_close.sav
nombre = 'model_close'
#k_clusters(csv, ndias, n_clusters, nombre)

#%% Importamos model_close para futuro uso.
model_close = pickle.load(open('model_close.sav','rb'))

#%% Se crea vector de toma de decisiones para futuro uso
Ud = np.random.randint(-1,2,len(model_close)**len(ndias))

#%%############################################################################
######################  Simulación para optimización ##########################
###############################################################################

#%% Simulamos
Vp = simulacion(csv,ndias,model_close,Ud)


#%% Calculamos y graficamos algunos datos sobre el desempeño general
Rend =  np.diff(Vp) / Vp[:,:-1] #Rendimientos diarios.
Port = Rend.mean(axis=0) #Creamos un portafolio con la misma cantidad de dinero en cada activo. 
Mean = Port.mean() #Se calcula el rendimiento esperado del portafolio y la estrategia aplicada
Std = Port.std() #Se calcula la volatilidad diaria
plt.scatter((Std*252)**0.5,Mean*252) #Se grafica la volatilidad anual contra el rendimiento anual. 

#%%############################################################################
#########################  Simulación para Graficar ###########################
###############################################################################

#%% graficamos todas las simulaciones
Sim = grafico(csv,ndias,model_close,Ud)

#%%############################################################################
###################  Optimización por algorigmo genético ######################
###############################################################################

#%% Simulamos

l_vec = len(model_close)**len(ndias) # longitud de cada vector de toma de decisiones
n_vec = 32 # cantidad de vectores de toma de decisiones por generacion en potencias de 2. 
iteraciones = 5
C = 0.5
nombre = 'prueba'
#####genetico(func,csv,ndias,model_close,l_vec,l_dec,iteraciones,C)
genetico(simulacion,csv,ndias,model_close,l_vec,n_vec,iteraciones,C,nombre)

#%%
[punt,padres,hist_mean,hist_std,hist_cal,hist_padres] = pickle.load(open(nombre + '.sav','rb'))


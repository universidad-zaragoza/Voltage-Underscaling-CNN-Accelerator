#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle as pickle
import tensorflow as tf
import numpy as np
from Stats import WeightQuantization,IntroduceFaultsInWeights,GenerateFaultsList
from Nets import GetNeuralNetworkModel
from Training import GetDatasets
from Simulation import get_all_outputs
from Simulation import buffer_simulation, save_obj, load_obj
from datetime import datetime
import collections 
import pandas as pd
import random
from Simulation import save_obj, load_obj
from funciones import buffer_vectores
import collections
from datetime import datetime
import itertools
import matplotlib.pyplot as plt
import pathlib
import os, sys



##Ciclo para recorrer todo los ficheros e ir guardando las mácaras y las firecciones con nombres diferentes
##luego ejecutar el static_index

print('cambio')
buffer_size = 16777216



#
#ruta_bin = 'C:/Users/usuario/Desktop/MoRS/MoRS-master/Modelo3_col_8_0.57/index'
#ruta_bin = 'MoRS/Modelo3_col_8_0.51/index'
#ruta_bin = 'MoRS/Modelo3_mas_fallos_col_8_experimentos/index'
# directorio = pathlib.Path(ruta_bin)
#
# ficheros = [fichero.name for fichero in directorio.iterdir()]
# ficheros.sort()
# #
# fich=1
# for no_fichero, j in enumerate(ficheros):
#     #print('j',j)
#     print(str(j))
#
#     directorio = os.path.join(ruta_bin, j)
#     print(directorio)
#
#     mbuffer = np.array(['x']*(buffer_size))
#     print(len(mbuffer))
#     index = pd.read_excel(directorio)
#     #index=pd.read_excel('MoRS\index\index.xlsx')
#     np_array = index.values
#    # print(np_array)
#     count_fallos=0
#
#
#     for i,x in enumerate(mbuffer):
#         #print(x)
#         if i in np_array:
#             #print(i)
#             mbuffer[i]=random.randint(0, 1)
#             count_fallos += 1
#
#     dist = collections.Counter(mbuffer)
#         #print(mbuffer[i])
#     print(len(mbuffer))
#     print((mbuffer))
#     print(dist)
#     print('count_fallos',count_fallos)
#
#
#     mod=np.array(['x']*8)
#     union=np.concatenate([mod, mbuffer])
#     #union = union[:-8]
#     print('tanaño',len(union))
#
#     error_mask, locs = (buffer_vectores(union[0:buffer_size]))
#     #error_mask, locs = (buffer_vectores(mbuffer))
#
#     print('tamaño de error mask',len(error_mask))
#     print('tamaño de error locs',len(locs))
#
#     save_obj(error_mask,'MoRS/Modelo3_col_8_0.51/mask/error_mask_'+ str(fich))
#     save_obj(locs,'MoRS/Modelo3_col_8_0.51/mask/locs_'+str(fich))
#
#
#     DF_mask = pd.DataFrame(error_mask)
#     DF_locs = pd.DataFrame(locs)
#
#     Mask_locs = pd.concat([DF_mask, DF_locs], axis=1, join='outer')
#     Mask_locs.columns = ['error_mask','locs']
#     Mask_locs.to_excel('MoRS/Modelo3_col_8_0.51/Mask_locs/Mask_locs_' +str(fich) + '.xlsx', index=False)
#     fich= fich+1


## para ficheros que no caben en un excel
vol='0.60'
fich = 0
for i in range(10):

    with open('MoRS/Modelo3_col_8_'+ str(vol)+'/index/index_'+ str(fich)+'.pkl', 'rb') as f:
        # Carga el contenido del archivo en una variable como lista de Python
        index_lista = pickle.load(f)

# Convierte la lista de Python en un arreglo de NumPy
    index_buffer = np.array(index_lista)
    print(index_buffer)

    mbuffer = np.array(['x']*(buffer_size))
    #print(len(mbuffer))

    count_fallos=0


    for i,x in enumerate(mbuffer):

        if i in index_buffer:
        #print(i)
            mbuffer[i]=random.randint(0, 1)
            count_fallos += 1
            #print(count_fallos)

    dist = collections.Counter(mbuffer)
            #print(mbuffer[i])
    print(len(mbuffer))
    print((mbuffer))
    print(dist)
    #print('count_fallos',count_fallos)


    mod=np.array(['x']*8)
    union=np.concatenate([mod, mbuffer])
#union = union[:-8]
    #print('tanaño',len(union))

    error_mask, locs = (buffer_vectores(union[0:buffer_size]))
    #error_mask, locs = (buffer_vectores(mbuffer))

    print('tamaño de error mask',len(error_mask))
    print('tamaño de error locs',len(locs))

    save_obj(error_mask,'MoRS/Modelo3_col_8_'+ str(vol)+'/mask/error_mask_'+ str(fich))
    save_obj(locs,'MoRS/Modelo3_col_8_'+ str(vol)+'/mask/locs_'+str(fich))


    DF_mask = pd.DataFrame(error_mask)
    DF_locs = pd.DataFrame(locs)

    Mask_locs = pd.concat([DF_mask, DF_locs], axis=1, join='outer')
    Mask_locs.columns = ['error_mask','locs']
    #Mask_locs.to_excel('MoRS/Modelo3_col_8_0.51/Mask_locs/Mask_locs_' +str(fich) + '.xlsx', index=False)
    fich= fich+1

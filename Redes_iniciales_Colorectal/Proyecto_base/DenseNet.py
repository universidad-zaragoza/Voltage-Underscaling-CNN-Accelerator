#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

import tensorflow as tf
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Training import GetDatasets
from Nets  import GetNeuralNetworkModel
from Stats import WeightQuantization, ActivationStats, CheckAccuracyAndLoss, QuantizationEffect, GetReadAndWrites
from Simulation import buffer_simulation, save_obj, load_obj
from Stats import CheckAccuracyAndLoss
from FileAnalize import analize_file, analize_file_uno,analize_file_uno_ceros, save_file, load_file
from funciones import buffer_vectores
import collections
from datetime import datetime
import itertools
import pathlib
import os, sys
from openpyxl import Workbook
from openpyxl import load_workbook


tf.random.set_seed(1234)
np.random.seed(1234)


# In[ ]:





# a) Get Dataset

# In[2]:


train_batch_size = test_batch_size = 32

train_set,valid_set,test_set = GetDatasets('colorectal_histology',(80,5,15),(224,224), 8, train_batch_size, test_batch_size)


# a) Get Model

# In[3]:


import tensorflow_datasets as tfds
tfds.load('colorectal_histology')


# Se crea la red, sin activar la cuantización ni el efecto de envejecimiento
# Las dimensiones de entrada de la imagen (224,224), el número de clases (8) y el tamaño de los batches

# In[4]:


DenseNet   = GetNeuralNetworkModel('DenseNet',(224,224,3),8, quantization = False, aging_active=False)
loss      = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
DenseNet.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])


#  b) Load/Save Weigths

# In[5]:


cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'DenseNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir, 'Weights')
DenseNet.load_weights(wgt_dir)


# Evaluación del accuracy y loss de la red

# In[6]:


(OrigLoss,OrigAcc) = DenseNet.evaluate(test_set)
print(test_set)


# 2) Stats

#  Write/Read Stats¶

# Se identifican (manualmente) las capas procesadadas(Convoluciones, Full conectadas y Pooling) junto a las capas que contienen los resultados que se escribiran en el buffer (capas luego de la funcion de activacion y/o Normalizacion)

# In[197]:


for index,layer in enumerate(DenseNet.layers):
    print(index,layer.name)
print('Las capas 0,3,9,11,17,19,25,31,37,40,45 y 50  contienen la informacion para su procesamiento en MMU')
print('Las capas 2,8,10,16,18,24,30,36,38,44,49 y 53 contienen las activaciones que son escritas en memoria')


# Con el siguiente bloque obtenemos el número de lecturas y escrituras por posición de memoria tanto usando la estrategia de CNN Gating o sin usarla

# In[13]:


Indices=[0,4,11,12,16,(22,11),25,29,(35,24),38,42,(48,37),51,55,(61,50),64,68,(74,63),77,81,(87,76),90,94,97,99,103,(109,97),
        112,116,(122,111),125,129,(135,124),138,142,(148,137),151,155,(161,150),164,168,(174,163),177,181,(187,176),
        190,194,(200,189),203,207,(213,202),216,220,(226,215),229,233,(239,228),242,246,(252,241),255,259,262,264,268,(274,262),
        277,281,(287,276),290,294,(300,289),303,307,(313,302),316,320,(326,315),329,333,(339,328),342,346,(352,341),
        355,359,(365,354),368,372,(378,367),381,385,(391,380),394,398,(404,393),407,411,(417,406),420,424,(430,419),
        433,437,(443,432),446,450,(456,445),459,463,(469,458),472,476,(482,471),485,489,(495,484),498,502,(508,497),
        511,515,(521,510),524,528,(534,523),537,541,(547,536),550,554,(560,549),563,567,(573,562),576,580,583,585,589,(595,583),
        598,602,(608,597),611,615,(621,610),624,628,(634,623),637,641,(647,636),650,654,(660,649),663,667,(673,662),
        676,680,(686,675),689,693,(699,688),702,706,(712,701),715,719,(725,714),728,732,(738,727),741,745,(751,740),
        754,758,(764,753),767,771,(777,765),780,784,(790,779),793,797,800]
Data     = GetReadAndWrites(DenseNet,Indices,831744,10,CNN_gating=False,network_name='DenseNet')
stats    = {'Lecturas': Data['Reads'],'Escrituras': Data['Writes']}
Baseline_Acceses   = pd.DataFrame(stats).reset_index(drop=False)
Data     = GetReadAndWrites(DenseNet,Indices,831744,10,CNN_gating=True,network_name='DenseNet')
stats    = {'Lecturas': Data['Reads'],'Escrituras': Data['Writes']}
CNN_gating_Acceses   = pd.DataFrame(stats).reset_index(drop=False)
#save_obj(Baseline_Acceses,'Data/Acceses/DenseNet/Baseline')
#save_obj(CNN_gating_Acceses,'Data/Acceses/DenseNet/CNN_gating_Adj')


# Analizar la posibilidad de usar menos bits

# In[11]:


CheckAccuracyAndLoss('DenseNet', test_set, wgt_dir, act_frac_size = 12, act_int_size = 3, wgt_frac_size = 13, wgt_int_size = 2,
                    input_shape = (224,224,3), output_shape = 8, batch_size = test_batch_size);


#  c) Activation Stats

# Para la configuración anterior, se observa el valor medio,maximo,minimo y el ratio de saturación tanto de las activaciones procesadas dentro de la unidad matricial de multiplicacion como de las almacenadas en el buffer. Nota: el ultimo parametro indica el numero de iteraciones que se deben realizar hasta agotar el dataset, se obtiene como numero de imagenes del dataset dividido en el batch size.

# In[12]:


ActivationStats(DenseNet,test_set,12,3,24)


# 3) Buffer Simulation

#  a) Baseline

# Ahora para el Baseline simularemos el comportamiento de 1 buffer durante la inferencia de 3 imagenes (solo 3 como ejemplo), la red se crea ahora activando la cuantizacion pero no el envejecimiento. LI y AI son los definidos en el item 2) Stats

# ......estos ficheros ya están , solo cargar e interpretarlos

# Sin inyectar errores original

# In[51]:


# from copy import deepcopy


cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'DenseNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir,'Weights')


trainBatchSize = testBatchSize = 16
_,_,test_dataset = GetDatasets('colorectal_histology',(80,5,15),(224,224), 8, trainBatchSize, testBatchSize)


Accs_x= []



network_size   = 290400*16   # Tamaño del buffer (en bits)
num_of_samples = 1       # Numero de muestras de distintas configuraciones de fallos a testear por cada valor de Accs/Loss

n_bits_total = np.ceil(network_size).astype(int)    #numero de bits totales

buffer = np.array(['x']*(network_size))
print(buffer)
for index in range(0,num_of_samples):

    loss,acc_x   = CheckAccuracyAndLoss('DenseNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
                                            act_frac_size = 12, act_int_size = 3, wgt_frac_size = 13, wgt_int_size = 2,
                                            batch_size=testBatchSize, verbose = 0, aging_active = False, weights_faults = False)


    print(acc_x)



    Accs_x.append(acc_x)

print(str(n_bits_total)+' completada: ', datetime.now().strftime("%H:%M:%S"))
#save_obj(Accs_x,'Data/Errors/DenseNet/Colorectal Dataset/Accs_x')
#save_obj(Loss,'Data/Errors/DenseNet/Colorectal Dataset/Loss')


# # Analizando el fichero en su estado original

# In[38]:


## from copy import deepcopy



cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'DenseNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir,'Weights')


trainBatchSize = testBatchSize = 16
_,_,test_dataset = GetDatasets('colorectal_histology',(80,5,15),(224,224), 8, trainBatchSize, testBatchSize)


Accs = []
Accs_w = []
Accs_a_w = []
Accs_1 = []
Accs_w_1 = []
Accs_a_w_1 = []
buffer_size= 16777216
Accs_1_0 = []
Accs_w_1_0 = []
Accs_a_w_1_0 = []
buffer_size= 16777216


ruta_bin = r'Data/Fault Characterization/VC707/RawData'
#ruta_bin = 'Data/Fault Characterization/KC705_B/RowData/'
directorio = pathlib.Path(ruta_bin)



ficheros = [fichero.name for fichero in directorio.iterdir()]


for base, dirs, files in os.walk(ruta_bin):
    print(files)

    for i, j in enumerate(files):
        #
        directorio = os.path.join(ruta_bin, j)
        print(directorio)
        buffer = (analize_file(directorio, buffer_size))
        error_mask, locs = (buffer_vectores(buffer))



# for i, j in enumerate(ficheros):
#     directorio= os.path.join(ruta_bin, j)
#     buffer= (analize_file(directorio, buffer_size))
#     error_mask, locs = (buffer_vectores(buffer))
#     print(directorio)

    loss,acc   = CheckAccuracyAndLoss('DenseNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
                                            act_frac_size = 12, act_int_size = 3, wgt_frac_size = 13, wgt_int_size = 2,
                                            batch_size=testBatchSize, verbose = 0, aging_active = True, weights_faults = False,
                                            faulty_addresses = locs, masked_faults = error_mask)

    loss,acc_w   = CheckAccuracyAndLoss('DenseNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
                                            act_frac_size = 12, act_int_size = 3, wgt_frac_size = 13, wgt_int_size = 2,
                                            batch_size=testBatchSize, verbose = 0, aging_active = False, weights_faults = True,
                                            faulty_addresses = locs, masked_faults = error_mask)

    loss,acc_a_w   = CheckAccuracyAndLoss('DenseNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
                                            act_frac_size = 12, act_int_size = 3, wgt_frac_size = 13, wgt_int_size = 2,
                                            batch_size=testBatchSize, verbose = 0, aging_active = True, weights_faults = True,
                                            faulty_addresses = locs, masked_faults = error_mask)


    Accs.append(acc)
    Accs_w.append(acc_w)
    Accs_a_w.append(acc_a_w)

    print('acc_0:', Accs)
    print('acc_w0:',Accs_w)
    print('acc_a_w_0:',Accs_a_w)



Acc=pd.DataFrame(Accs)
Acc_w =pd.DataFrame(Accs_w)
Acc_a_w =pd.DataFrame(Accs_a_w)
#buf_cero = pd.concat([Acc,Acc_w, Acc_a_w], axis=1, join='outer')
#buf_cero.columns =['Acc_cero', 'A_w_cero', 'Acc_a_w_cero']
#buf_cero.to_excel('resultado.xlsx', sheet_name='ceros_707', index=False)


Acc.head()


print(str()+' operación completada: ', datetime.now().strftime("%H:%M:%S"))
#lsave_file(Accs,'Data/Fault Characterization/Accs')

#save_obj(Loss,'Data/Errors/DenseNet/Colorectal Dataset/Loss')


# # Inyectando errores en 1

# In[42]:



ruta_bin = r'Data/Fault Characterization/VC707/RawData'
#ruta_bin = 'Data/Fault Characterization/KC705_B/RowData/'
directorio = pathlib.Path(ruta_bin)


ficheros = [fichero.name for fichero in directorio.iterdir()]
print(directorio)

for i, j in enumerate(ficheros):
    directorio= os.path.join(ruta_bin, j)
    buffer= (analize_file_uno(directorio, buffer_size))
    error_mask, locs = (buffer_vectores(buffer))
    print(directorio)

    loss,acc_1   = CheckAccuracyAndLoss('DenseNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
                                            act_frac_size = 12, act_int_size = 3, wgt_frac_size = 13, wgt_int_size = 2,
                                            batch_size=testBatchSize, verbose = 0, aging_active = True, weights_faults = False,
                                            faulty_addresses = locs, masked_faults = error_mask)

    loss,acc_w_1   = CheckAccuracyAndLoss('DenseNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
                                            act_frac_size = 12, act_int_size = 3, wgt_frac_size = 13, wgt_int_size = 2,
                                            batch_size=testBatchSize, verbose = 0, aging_active = False, weights_faults = True,
                                            faulty_addresses = locs, masked_faults = error_mask)

    loss,acc_a_w_1   = CheckAccuracyAndLoss('DenseNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
                                            act_frac_size = 12, act_int_size = 3, wgt_frac_size = 13, wgt_int_size = 2,
                                            batch_size=testBatchSize, verbose = 0, aging_active = True, weights_faults = True,
                                            faulty_addresses = locs, masked_faults = error_mask)


    Accs_1.append(acc_1)
    Accs_w_1.append(acc_w_1)
    Accs_a_w_1.append(acc_a_w_1)

    print('acc_1:', Accs_1)
    print('acc_w_1:', Accs_w_1)
    print('acc_a_w_1:', Accs_a_w_1)


Acc_1=pd.DataFrame(Accs_1)
Acc_w_1 =pd.DataFrame(Accs_w_1)
Acc_a_w_1 =pd.DataFrame(Accs_a_w_1)
#buf_cero = pd.concat([Acc_1,Acc_w_1, Acc_a_w_1], axis=1, join='outer')
#buf_cero.columns =['Acc_uno', 'A_w_uno', 'Acc_a_w_uno']
#buf_cero.to_excel('resultado.xlsx', sheet_name='unos_707', index=False)

#buf_cero = pd.concat([Acc,Acc_w, Acc_a_w], axis=1, join='outer')
#buf_cero.columns =['Acc_cero', 'A_w_cero', 'Acc_a_w_cero']
#buf_cero.to_excel('resultado.xlsx', sheet_name='ceros_707', index=False)



Acc.head()









print(str()+' operación completada: ', datetime.now().strftime("%H:%M:%S"))
save_file(Accs_1,'Data/Fault Characterization/Accs')

#save_obj(Loss,'Data/Errors/DenseNet/Colorectal Dataset/Loss')


# # Inyectando errores aleatorios ceros y unos

# In[44]:




ruta_bin = r'Data/Fault Characterization/VC707/RawData'
#ruta_bin = 'Data/Fault Characterization/KC705_B/RowData/'
directorio = pathlib.Path(ruta_bin)


ficheros = [fichero.name for fichero in directorio.iterdir()]

for i, j in enumerate(ficheros):
    print(ficheros)
    directorio= os.path.join(ruta_bin, j)
    buffer= (analize_file_uno_ceros(directorio, buffer_size))
    error_mask, locs = (buffer_vectores(buffer))
    print(directorio)

    loss,acc_1_0   = CheckAccuracyAndLoss('DenseNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
                                            act_frac_size = 12, act_int_size = 3, wgt_frac_size = 13, wgt_int_size = 2,
                                            batch_size=testBatchSize, verbose = 0, aging_active = True, weights_faults = False,
                                            faulty_addresses = locs, masked_faults = error_mask)

    loss,acc_w_1_0   = CheckAccuracyAndLoss('DenseNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
                                            act_frac_size = 12, act_int_size = 3, wgt_frac_size = 13, wgt_int_size = 2,
                                            batch_size=testBatchSize, verbose = 0, aging_active = False, weights_faults = True,
                                            faulty_addresses = locs, masked_faults = error_mask)

    loss,acc_a_w_1_0   = CheckAccuracyAndLoss('DenseNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
                                            act_frac_size = 12, act_int_size = 3, wgt_frac_size = 13, wgt_int_size = 2,
                                            batch_size=testBatchSize, verbose = 0, aging_active = True, weights_faults = True,
                                            faulty_addresses = locs, masked_faults = error_mask)


    Accs_1_0.append(acc_1_0)
    Accs_w_1_0.append(acc_w_1_0)
    Accs_a_w_1_0.append(acc_a_w_1_0)

    print('acc_1_0:', Accs_1_0)
    print('acc_w_1_0:', Accs_w_1_0)
    print('acc_a_w_1_0:', Accs_a_w_1_0)


Acc_1_0=pd.DataFrame(Accs_1_0)
Acc_w_1_0 =pd.DataFrame(Accs_w_1_0)
Acc_a_w_1_0 =pd.DataFrame(Accs_a_w_1_0)
buf_cero = pd.concat([Acc,Acc_w, Acc_a_w,Acc_1,Acc_w_1, Acc_a_w_1,Acc_1_0,Acc_w_1_0, Acc_a_w_1_0], axis=1, join='outer')
buf_cero.columns =['Acc_cero', 'A_w_cero', 'Acc_a_w_cero', 'Acc_uno', 'A_w_uno', 'Acc_a_w_uno' ,'Acc_uno_cero', 'A_w_uno_cero', 'Acc_a_w_uno_cero']
buf_cero.to_excel('resultado_DenseNet_vecinos.xlsx', sheet_name='fichero_707', index=False)

print(buf_cero)

#buf_cero = pd.concat([Acc_1,Acc_w_1, Acc_a_w_1], axis=1, join='outer')
#buf_cero.columns =['Acc_uno', 'A_w_uno', 'Acc_a_w_uno']
#buf_cero.to_excel('resultado.xlsx', sheet_name='unos_707', index=False)

#buf_cero = pd.concat([Acc,Acc_w, Acc_a_w], axis=1, join='outer')
#buf_cero.columns =['Acc_cero', 'A_w_cero', 'Acc_a_w_cero']
#buf_cero.to_excel('resultado.xlsx', sheet_name='ceros_707', index=False)
Acc.head()









print(str()+' operación completada: ', datetime.now().strftime("%H:%M:%S"))
#save_file(Accs_1,'Data/Fault Characterization/Accs')

#save_obj(Loss,'Data/Errors/DenseNet/Colorectal Dataset/Loss')


# In[ ]:





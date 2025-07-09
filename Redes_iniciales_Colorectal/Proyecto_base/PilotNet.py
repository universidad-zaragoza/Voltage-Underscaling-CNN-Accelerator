#!/usr/bin/env python
# coding: utf-8

# In[20]:
#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')



import tensorflow as tf
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Training import GetPilotNetDataset
from Training import GetDatasets
from Nets_original  import GetNeuralNetworkModel
from Stats import WeightQuantization, ActivationStats, CheckAccuracyAndLoss, QuantizationEffect, GetReadAndWrites
from datetime import datetime
from FileAnalize import analize_file, analize_file_uno,analize_file_uno_ceros, save_file, load_file
from funciones import buffer_vectores
import itertools
import pathlib
import os, sys
from openpyxl import Workbook
from openpyxl import load_workbook
from Simulation import buffer_simulation, save_obj, load_obj


tf.random.set_seed(1234)
np.random.seed(1234)


# # 1) Training

# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; a) Get Dataset

# In[21]:


trainbatch_size = test_batch_size = 32
train_set,valid_set,test_set = GetPilotNetDataset(csv_dir='Data/Datos/driving_log.csv', train_batch_size=32, test_batch_size=32)


# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; b) Get Model

# In[22]:


PilotNet = GetNeuralNetworkModel('PilotNet',(160,320,3),1,quantization = False,aging_active = False)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
# Compile Model
PilotNet.compile(optimizer=optimizer, loss='mse')


# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; c) Training

# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; d) Save/Load Weigths

# In[23]:


cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'PilotNet')
wgt_dir = os.path.join(wgt_dir,'Weights')
PilotNet.load_weights(wgt_dir)



OrigLoss = PilotNet.evaluate(test_set)




# # 2) Stats

# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  e) Activation Stats

# In[9]:





# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  f) Write/Read Stats

# In[19]:


Indices = [5,6,10,14,18,22,28,32,36,40,44]
#Data    = GetReadAndWrites(PilotNet,Indices,72912,150,CNN_gating=False)
#stats   = {'Lecturas': Data['Reads'],'Escrituras': Data['Writes']}
#Baseline_Acceses   = pd.DataFrame(stats).reset_index(drop=False)
Data   = GetReadAndWrites(PilotNet,Indices,72912,150,CNN_gating=True)
stats    = {'Lecturas': Data['Reads'],'Escrituras': Data['Writes']}
CNN_gating_Acceses = pd.DataFrame(stats).reset_index(drop=False)
#save_obj(Baseline_Acceses,'Data/Acceses/PilotNet/Baseline')
#save_obj(CNN_gating_Acceses,'Data/Acceses/PilotNet/CNN_gating_Adj')


# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; a) Starting Point

# In[24]:


CheckAccuracyAndLoss('PilotNet', test_set, wgt_dir, act_frac_size = 15, act_int_size = 0, wgt_frac_size = 15, wgt_int_size = 0, 
                    input_shape = (160,320,3), output_shape = 1, batch_size = test_batch_size);


# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; b) Number of bits analysis

# # 4) Error Injection

# In[25]:

ActivationStats(PilotNet,test_set,15,0,107)



trainBatchSize = testBatchSize = 32
_,_,test_dataset = GetPilotNetDataset(csv_dir='Data/Datos/driving_log.csv', train_batch_size=32, test_batch_size=32)

Loss_x= []



network_size   = 290400*16   # Tamaño del buffer (en bits)
num_of_samples = 1       # Numero de muestras de distintas configuraciones de fallos a testear por cada valor de Accs/Loss

n_bits_total = np.ceil(network_size).astype(int)    #numero de bits totales

buffer = np.array(['x']*(network_size))
print(buffer)
for index in range(0,num_of_samples):

    loss_x   = CheckAccuracyAndLoss('PilotNet', test_dataset, wgt_dir, act_frac_size = 15, act_int_size = 0, wgt_frac_size = 15, wgt_int_size = 0,
                                        input_shape =(160,320,3),output_shape=1, batch_size=testBatchSize, verbose = 0, aging_active = False, weights_faults = False)


    print(loss_x)



    Loss_x.append(loss_x)

print(str(n_bits_total)+' completada: ', datetime.now().strftime("%H:%M:%S"))
#save_obj(Accs_x,'Data/Errors/AlexNet/Colorectal Dataset/Accs_x')
#save_obj(Loss,'Data/Errors/AlexNet/Colorectal Dataset/Loss')





buffer_size= 16777216
Loss = []
Loss_w = []
Loss_a_w = []
Loss_1 = []
Loss_w_1 = []
Loss_a_w_1 = []
Loss_1_0 = []
Loss_w_1_0 = []
Loss_a_w_1_0 = []






ruta_bin = 'Data/Fault Characterization/VC707/RawData'
#ruta_bin = 'Data/Fault Characterization/KC705_B/RowData/'
directorio = pathlib.Path(ruta_bin)


ficheros = [fichero.name for fichero in directorio.iterdir()]
ficheros.sort()

for i, j in enumerate(ficheros):
    directorio= os.path.join(ruta_bin, j)
    buffer= (analize_file(directorio, buffer_size))
    error_mask, locs = (buffer_vectores(buffer))
    print(directorio)

    acc,loss   = CheckAccuracyAndLoss('PilotNet', test_dataset, wgt_dir, output_shape=1,  input_shape = (160,320,3),
                                            act_frac_size = 15, act_int_size = 0, wgt_frac_size = 15, wgt_int_size = 0,
                                            batch_size=testBatchSize, verbose = 0, aging_active = True, weights_faults = False,
                                            faulty_addresses = locs, masked_faults = error_mask)

   
    acc_w,loss_w   = CheckAccuracyAndLoss('PilotNet', test_dataset, wgt_dir, output_shape=1,  input_shape = (160,320,3),
                                            act_frac_size = 15, act_int_size = 0, wgt_frac_size = 15, wgt_int_size = 0,
                                            batch_size=testBatchSize, verbose = 0, aging_active = False, weights_faults = True,
                                            faulty_addresses = locs, masked_faults = error_mask)
   
    acc_a_w,loss_a_w   = CheckAccuracyAndLoss('PilotNet', test_dataset, wgt_dir, output_shape=1,  input_shape = (160,320,3),
                                            act_frac_size = 15, act_int_size = 0, wgt_frac_size = 15, wgt_int_size = 0,
                                            batch_size=testBatchSize, verbose = 0, aging_active = True, weights_faults = True,
                                            faulty_addresses = locs, masked_faults = error_mask)

    print(loss)
    print(loss_w)
    print(loss_a_w)
    Loss.append(loss)
    Loss_w.append(loss_w)
    Loss_a_w.append(loss_a_w)


Loss=pd.DataFrame(Loss)
Loss_w =pd.DataFrame(Loss_w)
Loss_a_w =pd.DataFrame(Loss_a_w)
#buf_cero = pd.concat([Acc,Acc_w, Acc_a_w], axis=1, join='outer')
#buf_cero.columns =['Acc_cero', 'A_w_cero', 'Acc_a_w_cero']
#buf_cero.to_excel('resultado.xlsx', sheet_name='ceros_707', index=False)





# In[ ]:


print(str()+' operación completada: ', datetime.now().strftime("%H:%M:%S"))


# In[ ]:



ruta_bin = 'Data/Fault Characterization/VC707/RawData'
#ruta_bin = 'Data/Fault Characterization/KC705_B/RowData/'
directorio = pathlib.Path(ruta_bin)


ficheros = [fichero.name for fichero in directorio.iterdir()]
ficheros.sort()
print(directorio)

for i, j in enumerate(ficheros):
    directorio= os.path.join(ruta_bin, j)
    buffer= (analize_file_uno(directorio, buffer_size))
    error_mask, locs = (buffer_vectores(buffer))
    print(directorio)

    acc_1,loss_1   = CheckAccuracyAndLoss('PilotNet', test_dataset, wgt_dir, output_shape=1,  input_shape = (160,320,3),
                                            act_frac_size = 15, act_int_size = 0, wgt_frac_size = 15, wgt_int_size = 0,
                                            batch_size=testBatchSize, verbose = 0, aging_active = True, weights_faults = False,
                                            faulty_addresses = locs, masked_faults = error_mask)

   
    acc_w_1,loss_w_1   = CheckAccuracyAndLoss('PilotNet', test_dataset, wgt_dir, output_shape=1,  input_shape = (160,320,3),
                                            act_frac_size = 15, act_int_size = 0, wgt_frac_size = 15, wgt_int_size = 0,
                                            batch_size=testBatchSize, verbose = 0, aging_active = False, weights_faults = True,
                                            faulty_addresses = locs, masked_faults = error_mask)
   
    acc_a_w_1,loss_a_w_1   = CheckAccuracyAndLoss('PilotNet', test_dataset, wgt_dir, output_shape=1,  input_shape = (160,320,3),
                                            act_frac_size = 15, act_int_size = 0, wgt_frac_size = 15, wgt_int_size = 0,
                                            batch_size=testBatchSize, verbose = 0, aging_active = True, weights_faults = True,
                                            faulty_addresses = locs, masked_faults = error_mask)

    print(loss_1)
    print(loss_w_1)
    print(loss_a_w_1)
    Loss_1.append(loss_1)
    Loss_w_1.append(loss_w_1)
    Loss_a_w_1.append(loss_a_w_1)


Loss_1=pd.DataFrame(Loss_1)
Loss_w_1 =pd.DataFrame(Loss_w_1)
Loss_a_w_1 =pd.DataFrame(Loss_a_w_1)
#buf_cero = pd.concat([Acc_1,Acc_w_1, Acc_a_w_1], axis=1, join='outer')
#buf_cero.columns =['Acc_uno', 'A_w_uno', 'Acc_a_w_uno']
#buf_cero.to_excel('resultado.xlsx', sheet_name='unos_707', index=False)

#buf_cero = pd.concat([Acc,Acc_w, Acc_a_w], axis=1, join='outer')
#buf_cero.columns =['Acc_cero', 'A_w_cero', 'Acc_a_w_cero']
#buf_cero.to_excel('resultado.xlsx', sheet_name='ceros_707', index=False)





print(str()+' operación completada: ', datetime.now().strftime("%H:%M:%S"))
#save_file(Loss_1,'Data/Fault Characterization/Loss')





ruta_bin = 'Data/Fault Characterization/VC707/RawData'
#ruta_bin = 'Data/Fault Characterization/KC705_B/RowData/'
directorio = pathlib.Path(ruta_bin)

ficheros = [fichero.name for fichero in directorio.iterdir()]
ficheros.sort()
print(directorio)

for i, j in enumerate(ficheros):
    directorio= os.path.join(ruta_bin, j)
    buffer= (analize_file_uno_ceros(directorio, buffer_size))
    error_mask, locs = (buffer_vectores(buffer))
    print(directorio)

    acc_1_0,loss_1_0  = CheckAccuracyAndLoss('PilotNet', test_dataset, wgt_dir, output_shape=1,  input_shape = (160,320,3),
                                            act_frac_size = 15, act_int_size = 0, wgt_frac_size = 15, wgt_int_size = 0,
                                            batch_size=testBatchSize, verbose = 0, aging_active = True, weights_faults = False,
                                            faulty_addresses = locs, masked_faults = error_mask)

   
    acc_w_1_0,loss_w_1_0   = CheckAccuracyAndLoss('PilotNet', test_dataset, wgt_dir, output_shape=1,  input_shape = (160,320,3),
                                            act_frac_size = 15, act_int_size = 0, wgt_frac_size = 15, wgt_int_size = 0,
                                            batch_size=testBatchSize, verbose = 0, aging_active = False, weights_faults = True,
                                            faulty_addresses = locs, masked_faults = error_mask)
   
    acc_a_w_1_0,loss_a_w_1_0   = CheckAccuracyAndLoss('PilotNet', test_dataset, wgt_dir, output_shape=1,  input_shape = (160,320,3),
                                            act_frac_size = 15, act_int_size = 0, wgt_frac_size = 15, wgt_int_size = 0,
                                            batch_size=testBatchSize, verbose = 0, aging_active = True, weights_faults = True,
                                            faulty_addresses = locs, masked_faults = error_mask)
    print(loss_1_0)
    print(loss_w_1_0)
    print(loss_a_w_1_0)
    Loss_1_0.append(loss_1_0)
    Loss_w_1_0.append(loss_w_1_0)
    Loss_a_w_1_0.append(loss_a_w_1_0)

Arquit=pd.DataFrame(ficheros)
Loss_1_0=pd.DataFrame(Loss_1_0)
Loss_w_1_0 =pd.DataFrame(Loss_w_1_0)
Loss_a_w_1_0 =pd.DataFrame(Loss_a_w_1_0)
buf_cero = pd.concat([Arquit,Loss,Loss_w, Loss_a_w,Loss_1,Loss_w_1, Loss_a_w_1,Loss_1_0,Loss_w_1_0, Loss_a_w_1_0], axis=1, join='outer')
buf_cero.columns =['Voltajes','L_cero', 'L_w_cero', 'L_a_w_cero', 'L_uno', 'L_w_uno', 'L_a_w_uno' ,'L_uno_cero', 'L_w_uno_cero', 'L_a_w_uno_cero']
buf_cero.to_excel('resultado_PiloNet_vecinos.xlsx', sheet_name='fichero_705', index=False)


print(str()+' operación completada: ', datetime.now().strftime("%H:%M:%S"))

print(buf_cero)
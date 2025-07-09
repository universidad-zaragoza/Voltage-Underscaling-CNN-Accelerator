#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os
import pickle as pickle

import tensorflow as tf

import numpy as np
#from Stats_original import WeightQuantization,IntroduceFaultsInWeights,GenerateFaultsList,CheckAccuracyAndLoss
from Stats_original import WeightQuantization,IntroduceFaultsInWeights,GenerateFaultsList,CheckAccuracyAndLoss
#from Nets_original import GetNeuralNetworkModel
from Nets import GetNeuralNetworkModel
from Training import GetPilotNetDataset
from Training import GetDatasets
from Simulation import get_all_outputs
from Simulation import buffer_simulation, save_obj, load_obj
from datetime import datetime
import collections 
import pandas as pd
import os, sys
import pathlib


# In[14]:
#Acc_net: Desde una ruta definida con los ficgeros de error_mask y locs ya guardados con anterioridad para los voltajes analizados 
#los recorre y calcula el acc por cada voltaje para cada red de las estudiadas, finalmente los huarda en un documento excel para un
# futuro análisis más detallado, si se desea hacer para otras máscaras de error basta con cambiar las direcciones donde se encuentren


#locs  = load_obj('Data/Fault Characterization/error_mask_0/vc_707/locs_060')
#error_mask = load_obj('Data/Fault Characterization/error_mask_0/vc_707/error_mask_060')


# # PilotNet

# In[ ]:


cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'PilotNet')
wgt_dir = os.path.join(wgt_dir,'Weights')


Accs_P=[]

ttrainBatchSize = testBatchSize = 32
__,_,test_dataset = GetPilotNetDataset(csv_dir='Data/Datos/driving_log.csv', train_batch_size=32, test_batch_size=32)


# ruta_bin = 'Data/Fault Characterization'
# #ruta_bin = 'Data/Fault Characterization/KC705_B/RowData/'
# directorio = pathlib.Path(ruta_bin)
#
# ficheros = [fichero.name for fichero in directorio.iterdir()]
voltaj=('0.54','0.55','0.56','0.57','0.58','0.59','0.60')
Voltajes=pd.DataFrame(voltaj)
# ficheros.sort()

vol=54
for i in range(7):
    locs = load_obj('Data/Fault Characterization/error_mask_0/vc_707/locs_0' + str(vol))
    error_mask = load_obj('Data/Fault Characterization/error_mask_0/vc_707/error_mask_0' + str(vol))

    vol = vol + 1
    print(i)
    print(vol)
    print(len(locs))

    acc,loss   = CheckAccuracyAndLoss('PilotNet', test_dataset, wgt_dir, output_shape=1,  input_shape = (160,320,3),
                                            act_frac_size = 15, act_int_size = 0, wgt_frac_size = 15, wgt_int_size = 0,
                                            batch_size=testBatchSize, verbose = 0, aging_active = True, weights_faults = False,
                                            faulty_addresses = locs, masked_faults = error_mask)
    Accs_P.append(loss)
    #Acc_PilotNet=pd.DataFrame(Accs_P)

# # AlexNet

# In[24]:


    cwd = os.getcwd()
    wgt_dir = os.path.join(cwd, 'Data')
    wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
    wgt_dir = os.path.join(wgt_dir, 'AlexNet')
    wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
    wgt_dir = os.path.join(wgt_dir, 'Weights')
    
    Accs_A=[]

    trainBatchSize = testBatchSize = 16
    _,_,test_dataset = GetDatasets('colorectal_histology',(80,5,15),(227,227), 8, trainBatchSize, testBatchSize)

    # ruta_bin = 'Data/Fault Characterization'
    # #ruta_bin = 'Data/Fault Characterization/KC705_B/RowData/'
    # directorio = pathlib.Path(ruta_bin)
    #
    # ficheros = [fichero.name for fichero in directorio.iterdir()]
    #
    # ficheros.sort()

    #vol=54
    # for i in range(1):
    #     locs = load_obj('Data/Fault Characterization/error_mask y locs_buffer_act_vc-707/locs_0' + str(vol))
    #     error_mask = load_obj('Data/Fault Characterization/error_mask y locs_buffer_act_vc-707/error_mask_0' + str(vol))

    # vol=vol+1
    # print(i)
    # print(vol)
    # print(len(locs))
    #





    loss,acc   = CheckAccuracyAndLoss('AlexNet', test_dataset, wgt_dir, output_shape=8, input_shape = (227,227,3),
                                                    act_frac_size = 11, act_int_size = 4, wgt_frac_size = 11, wgt_int_size = 4,
                                                    batch_size=testBatchSize, verbose = 0, aging_active = True, weights_faults = False,
                                                    faulty_addresses = locs, masked_faults = error_mask)

    Accs_A.append(acc)
    #Acc_AlexNet=pd.DataFrame(Accs_A)


    # In[22]:





    # # SqueezeNet

    # In[15]:



        # Directorio de los pesos
    cwd = os.getcwd()
    wgt_dir = os.path.join(cwd, 'Data')
    wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
    wgt_dir = os.path.join(wgt_dir, 'SqueezeNet')
    wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
    wgt_dir = os.path.join(wgt_dir,'Weights')
    Accs_S=[]


    trainBatchSize = testBatchSize = 16
    _,_,test_dataset = GetDatasets('colorectal_histology',(80,5,15),(224,224), 8, trainBatchSize, testBatchSize)

    # ruta_bin = 'Data/Fault Characterization'
    # #ruta_bin = 'Data/Fault Characterization/KC705_B/RowData/'
    # directorio = pathlib.Path(ruta_bin)
    #
    # ficheros = [fichero.name for fichero in directorio.iterdir()]
    #
    # ficheros.sort()

    # vol=54
    # for i in range(1):
    #     locs = load_obj('Data/Fault Characterization/error_mask y locs_buffer_act_vc-707/locs_0' + str(vol))
    #     error_mask = load_obj('Data/Fault Characterization/error_mask y locs_buffer_act_vc-707/error_mask_0' + str(vol))
    #     vol=vol+1
    #     print(i)
    #     print(vol)
    #     print(len(locs))
    #


    loss,acc   = CheckAccuracyAndLoss('SqueezeNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
                                                    act_frac_size = 9, act_int_size = 6, wgt_frac_size = 15, wgt_int_size = 0,
                                                    batch_size=testBatchSize, verbose = 0, aging_active = True, weights_faults = False,
                                                    faulty_addresses = locs, masked_faults = error_mask)

    Accs_S.append(acc)
   # Acc_SqueezeNet=pd.DataFrame(Accs_S)


    # In[16]:





    # # DenseNet

    # In[21]:



    # Directorio de los pesos
    cwd = os.getcwd()
    wgt_dir = os.path.join(cwd, 'Data')
    wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
    wgt_dir = os.path.join(wgt_dir, 'DenseNet')
    wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
    wgt_dir = os.path.join(wgt_dir,'Weights')

    Accs_D=[]

    trainBatchSize = testBatchSize = 16
    _,_,test_dataset = GetDatasets('colorectal_histology',(80,5,15),(224,224), 8, trainBatchSize, testBatchSize)
    # ruta_bin = 'Data/Fault Characterization'
    # #ruta_bin = 'Data/Fault Characterization/KC705_B/RowData/'
    # directorio = pathlib.Path(ruta_bin)
    #
    # ficheros = [fichero.name for fichero in directorio.iterdir()]
    #
    # ficheros.sort()

    # vol=54
    # for i in range(1):
    #     locs = load_obj('Data/Fault Characterization/error_mask y locs_buffer_act_vc-707/locs_0' + str(vol))
    #     error_mask = load_obj('Data/Fault Characterization/error_mask y locs_buffer_act_vc-707/error_mask_0' + str(vol))
    #
    #     vol=vol+1
    #     print(i)
    #     print(vol)
    #     print(len(locs))
    #


    loss,acc   = CheckAccuracyAndLoss('DenseNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
                                                    act_frac_size = 12, act_int_size = 3, wgt_frac_size = 13, wgt_int_size = 2,
                                                    batch_size=testBatchSize, verbose = 0, aging_active = True, weights_faults = False,
                                                    faulty_addresses = locs, masked_faults = error_mask)

    Accs_D.append(acc)
    # Acc_DenseNet=pd.DataFrame(Accs_D)


    # In[22]:





    # # MobileNet

    # In[25]:



    # Directorio de los pesos
    cwd = os.getcwd()
    wgt_dir = os.path.join(cwd, 'Data')
    wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
    wgt_dir = os.path.join(wgt_dir, 'MobileNet')
    wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
    wgt_dir = os.path.join(wgt_dir,'Weights')
    Accs_M=[]


    trainBatchSize = testBatchSize = 16
    _,_,test_dataset = GetDatasets('colorectal_histology',(80,5,15),(224,224), 8, trainBatchSize, testBatchSize)

    #ruta_bin = 'Data/Fault Characterization/error_mask_0/vc_707'
    #ruta_bin = 'Data/Fault Characterization/KC705_B/RowData/'
    # directorio = pathlib.Path(ruta_bin)
    #
    # ficheros = [fichero.name for fichero in directorio.iterdir()]
    # voltaj=('0.54','0.55','0.56','0.57','0.58','0.59','0.60')
    # Voltajes=pd.DataFrame(voltaj)
    #
    # ficheros.sort()

    #vol=54
    # for i in range(1):
    #     locs = load_obj('Data/Fault Characterization/error_mask y locs_buffer_act_vc-707/locs_0'+str(vol))
    #     error_mask = load_obj('Data/Fault Characterization/error_mask y locs_buffer_act_vc-707/error_mask_0'+str(vol))
    #     #error_mask=load_obj('Data/Fault Characterization/error_mask_0x_30/vc_707/error_mask_0'+str(vol))
    #     #locs  = load_obj('Data/Fault Characterization/error_mask_0x_30/vc_707/locs_0'+ str(vol))
    #
    # vol=vol+1
    # print(i)
    # print(vol)
    # print(len(locs))




    loss,acc   = CheckAccuracyAndLoss('MobileNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
                                                    act_frac_size = 11, act_int_size = 4, wgt_frac_size = 14, wgt_int_size = 1,
                                                    batch_size=testBatchSize, verbose = 0, aging_active = True, weights_faults = False,
                                                    faulty_addresses = locs, masked_faults = error_mask)
    Accs_M.append(acc)
    # Acc_MobileNet=pd.DataFrame(Accs_M)


    # In[26]:





    # # VGG16

    # In[29]:




    # Directorio de los pesos
    cwd = os.getcwd()
    wgt_dir = os.path.join(cwd, 'Data')
    wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
    wgt_dir = os.path.join(wgt_dir, 'VGG16')
    wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
    wgt_dir = os.path.join(wgt_dir,'Weights')

    Accs_V=[]

    trainBatchSize = testBatchSize = 16
    _,_,test_dataset = GetDatasets('colorectal_histology',(80,5,15),(224,224), 8, trainBatchSize, testBatchSize)
    #ruta_bin = 'Data/Fault Characterization/error_mask_0/vc_707'
    #ruta_bin = 'Data/Fault Characterization/KC705_B/RowData/'
    # directorio = pathlib.Path(ruta_bin)
    #
    # ficheros = [fichero.name for fichero in directorio.iterdir()]
    #
    # ficheros.sort()

    #vol=54
    # for i in range(1):
    #     locs = load_obj('Data/Fault Characterization/error_mask y locs_buffer_act_vc-707/locs_0' + str(vol))
    #     error_mask = load_obj('Data/Fault Characterization/error_mask y locs_buffer_act_vc-707/error_mask_0' + str(vol))

    # vol=vol+1
    # print(i)
    # print(vol)
    # print(len(locs))
    #



    loss,acc   = CheckAccuracyAndLoss('VGG16', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
                                                    act_frac_size = 12, act_int_size = 3, wgt_frac_size = 15, wgt_int_size = 0,
                                                    batch_size=testBatchSize, verbose = 0, aging_active = True, weights_faults = False,
                                                    faulty_addresses = locs, masked_faults = error_mask)

    Accs_V.append(acc)
    # Acc_VGG16=pd.DataFrame(Accs_V)


    # # ZFNet

    # In[30]:




    # Directorio de los pesos
    cwd = os.getcwd()
    wgt_dir = os.path.join(cwd, 'Data')
    wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
    wgt_dir = os.path.join(wgt_dir, 'ZFNet')
    wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
    wgt_dir = os.path.join(wgt_dir,'Weights')

    Accs_Z=[]

    trainBatchSize = testBatchSize = 16
    _,_,test_dataset = GetDatasets('colorectal_histology',(80,5,15),(224,224), 8, trainBatchSize, testBatchSize)
    # #ruta_bin = 'Data/Fault Characterization/error_mask_0/vc_707'
    # #ruta_bin = 'Data/Fault Characterization/KC705_B/RowData/'
    # directorio = pathlib.Path(ruta_bin)
    #
    # ficheros = [fichero.name for fichero in directorio.iterdir()]
    #
    # ficheros.sort()

    # vol=54
    # for i in range(1):
    #     locs = load_obj('Data/Fault Characterization/error_mask y locs_buffer_act_vc-707/locs_0' + str(vol))
    #     error_mask = load_obj('Data/Fault Characterization/error_mask y locs_buffer_act_vc-707/error_mask_0' + str(vol))
    #
    #     vol=vol+1
    #     print(i)
    #     print(vol)
    #     print(len(locs))
    #


    loss,acc   = CheckAccuracyAndLoss('ZFNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
                                                    act_frac_size = 11, act_int_size = 4, wgt_frac_size = 15, wgt_int_size = 0,
                                                    batch_size=testBatchSize, verbose = 0, aging_active = True, weights_faults = False,
                                                    faulty_addresses = locs, masked_faults = error_mask)

    Accs_Z.append(acc)
Acc_PilotNet=pd.DataFrame(Accs_P)
Acc_AlexNet = pd.DataFrame(Accs_A)
Acc_SqueezeNet = pd.DataFrame(Accs_S)
Acc_DenseNet=pd.DataFrame(Accs_D)
Acc_MobileNet=pd.DataFrame(Accs_M)
Acc_VGG16=pd.DataFrame(Accs_V)
Acc_ZFNet=pd.DataFrame(Accs_Z)



buf_cero = pd.concat([Voltajes,Acc_AlexNet,Acc_SqueezeNet,Acc_DenseNet,Acc_MobileNet,Acc_VGG16,Acc_ZFNet,Acc_PilotNet], axis=1, join='outer')
buf_cero.columns =['Voltajes','Acc_Alex', 'Acc_Sque', 'Acc_Dense', 'Acc_Mobile', 'Acc_VGG16', 'Acc_ZFNet' ,'Acc_PilotNet']
buf_cero.to_excel('acc_total_error_base_round.xlsx', sheet_name='fichero_707', index=False)


print(str()+' operación completada: ', datetime.now().strftime("%H:%M:%S"))





#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os
import pickle as pickle
import tensorflow as tf
import numpy as np
from Stats_borrar import WeightQuantization,IntroduceFaultsInWeights,GenerateFaultsList,CheckAccuracyAndLoss
from Training import GetPilotNetDataset
from Training import GetDatasets
from Simulation import get_all_outputs
from Simulation import buffer_simulation, save_obj, load_obj
from datetime import datetime
import time
import collections 
import pandas as pd
import os, sys
import pathlib


# In[14]:
#Acc_net: Desde una ruta definida con los ficgeros de error_mask y locs ya guardados con anterioridad para los voltajes analizados 
#los recorre y calcula el acc por cada voltaje para cada red de las estudiadas, finalmente los huarda en un documento excel para un
# futuro análisis más detallado, si se desea hacer para otras máscaras de error basta con cambiar las direcciones donde se encuentren



# # PilotNet

# In[ ]:

voltaj=[54,55,56,57,58,59,60]
#Df_Vol=[54,56,58,60]
Voltajes=pd.DataFrame(voltaj)
Df_Vol=[]
paso = 1




# cwd = os.getcwd()
# wgt_dir = os.path.join(cwd, 'Data')
# wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
# wgt_dir = os.path.join(wgt_dir, 'PilotNet')
# wgt_dir = os.path.join(wgt_dir,'Weights')
#
#
# Accs_P=[]
# run_time_P=[]
#
# trainBatchSize = testBatchSize = 1
# __,_,test_dataset = GetPilotNetDataset(csv_dir='Data/Datos/driving_log.csv', train_batch_size=1, test_batch_size=1)
#
# print('test_dataset',test_dataset)
#
# #
# vol=voltaj[0]
# activation_aging = [True]*10
# for i in range(1):
#     inicio = time.time()
#
#     locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/flip_patch/byte_x/vc_707/locs_0' + str(vol))
#     error_mask = load_obj( 'Data/Fault Characterization/variante_mask_vc_707/flip_patch/byte_x/vc_707/error_mask_0' + str(vol))
#
#
#     Df_Vol.append(vol)
#     vol=vol + paso
#
#
#     acc,loss   = CheckAccuracyAndLoss('PilotNet', test_dataset, wgt_dir, output_shape=1,  input_shape = (160,320,3),
#                                             act_frac_size = 15, act_int_size = 0, wgt_frac_size = 15, wgt_int_size = 0,
#                                             batch_size=testBatchSize, verbose = 0, aging_active = activation_aging, weights_faults = False,
#                                             faulty_addresses = locs, masked_faults = error_mask)
#     Accs_P.append(loss)
#     fin = time.time()
#     time_run = fin - inicio
#     run_time_P.append(time_run)
# DF_run_time_p=pd.DataFrame(run_time_P)
# Acc_PilotNet=pd.DataFrame(Accs_P)



error_mask = load_obj(     'MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/error_mask_0' )
locs = load_obj(    'MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/locs_0')


cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'AlexNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir,'Weights')

Accs_A=[]
run_time_A=[]

trainBatchSize = testBatchSize = 1
_,_,test_dataset = GetDatasets('colorectal_histology',(80,5,15),(227,227), 8, trainBatchSize, testBatchSize)
print('pesos en true')



vol=voltaj[0]
activation_aging = [True]*11

for i in range(1):
    inicio = time.time()
    # error_mask = load_obj('Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/error_mask_0' + str(vol))
    # locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/locs_0' + str(vol))
    error_mask = load_obj(     'MoRS/Modelo3_mas_fallos_col_16_sinx_delante/F_Patch/error_mask_0' )
    locs = load_obj(    'MoRS/Modelo3_mas_fallos_col_16_sinx_delante/F_Patch/locs_0')
    vol=vol + paso
    # print(i)
    #print('voltaje',vol)
    print('tamaño de locs',len(locs))


    loss,acc   = CheckAccuracyAndLoss('AlexNet', test_dataset, wgt_dir, output_shape=8, input_shape = (227,227,3),
                                                act_frac_size = 11, act_int_size = 4, wgt_frac_size = 11, wgt_int_size = 4,
                                                batch_size=testBatchSize, verbose = 0, aging_active = activation_aging, weights_faults = False,
                                                faulty_addresses = locs, masked_faults = error_mask)

    Accs_A.append(acc)
    fin = time.time()
    time_run = fin - inicio
    run_time_A.append(time_run)
DF_run_time_a=pd.DataFrame(run_time_A)
Acc_AlexNet=pd.DataFrame(Accs_A)



print(str()+' operación completada AlexNet: ', datetime.now().strftime("%H:%M:%S"))





#     # Directorio de los pesos
cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'SqueezeNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir,'Weights')
Accs_S=[]
run_time_S=[]


trainBatchSize = testBatchSize = 1
_,_,test_dataset = GetDatasets('colorectal_histology',(80,5,15),(224,224), 8, trainBatchSize, testBatchSize)

# ruta_bin = 'Data/Fault Characterization/error_mask_0/vc_707'
# #ruta_bin = 'Data/Fault Characterization/KC705_B/RowData/'
# directorio = pathlib.Path(ruta_bin)
#
# ficheros = [fichero.name for fichero in directorio.iterdir()]
# #
# ficheros.sort()

vol=voltaj[0]
activation_aging = [True] * 22
for i in range(1):
    inicio = time.time()
    # locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/flip_patch/byte_x/vc_707/locs_0' + str(vol))
    # error_mask = load_obj( 'Data/Fault Characterization/variante_mask_vc_707/flip_patch/byte_x/vc_707/error_mask_0' + str(vol))

    vol = vol + paso
    #print('voltaje', vol)
    print('tamaño de locs', len(locs))



    loss,acc   = CheckAccuracyAndLoss('SqueezeNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
                                                act_frac_size = 9, act_int_size = 6, wgt_frac_size = 15, wgt_int_size = 0,
                                                batch_size=testBatchSize, verbose = 0, aging_active = activation_aging, weights_faults = False,
                                                faulty_addresses = locs, masked_faults = error_mask)

    Accs_S.append(acc)
    fin = time.time()
    time_run = fin - inicio
    run_time_S.append(time_run)
DF_run_time_s = pd.DataFrame(run_time_S)
Acc_SqueezeNet=pd.DataFrame(Accs_S)




print(str()+' operación completada SqueezeNet: ', datetime.now().strftime("%H:%M:%S"))




# # Directorio de los pesos
inicio = time.time()
cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'DenseNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir,'Weights')

Accs_D=[]
run_time_D=[]

trainBatchSize = testBatchSize = 1
_,_,test_dataset = GetDatasets('colorectal_histology',(80,5,15),(224,224), 8, trainBatchSize, testBatchSize)


vol=voltaj[0]
activation_aging = [True] * 188
for i in range(1):
    inicio = time.time()
    # locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/flip_patch/byte_x/vc_707/locs_0' + str(vol))
    # error_mask = load_obj( 'Data/Fault Characterization/variante_mask_vc_707/flip_patch/byte_x/vc_707/error_mask_0' + str(vol))

    vol = vol + paso

    loss,acc   = CheckAccuracyAndLoss('DenseNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
                                                act_frac_size = 12, act_int_size = 3, wgt_frac_size = 13, wgt_int_size = 2,
                                                batch_size=testBatchSize, verbose = 0, aging_active = activation_aging, weights_faults = False,
                                                faulty_addresses = locs, masked_faults = error_mask)

    Accs_D.append(acc)
    fin = time.time()
    time_run = fin - inicio
    run_time_D.append(time_run)
DF_run_time_d = pd.DataFrame(run_time_D)
Acc_DenseNet=pd.DataFrame(Accs_D)


# In[22]:

print(str()+' operación completada DenseNet: ', datetime.now().strftime("%H:%M:%S"))




# Directorio de los pesos
cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'MobileNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir,'Weights')
Accs_M=[]
run_time_M=[]


trainBatchSize = testBatchSize = 1
_,_,test_dataset = GetDatasets('colorectal_histology',(80,5,15),(224,224), 8, trainBatchSize, testBatchSize)



vol=voltaj[0]
activation_aging = [True]*29
for i in range(1):
    inicio = time.time()
    # locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/flip_patch/byte_x/vc_707/locs_0' + str(vol))
    # error_mask = load_obj( 'Data/Fault Characterization/variante_mask_vc_707/flip_patch/byte_x/vc_707/error_mask_0' + str(vol))

    vol = vol + paso


    loss,acc   = CheckAccuracyAndLoss('MobileNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
                                                act_frac_size = 11, act_int_size = 4, wgt_frac_size = 14, wgt_int_size = 1,
                                                batch_size=testBatchSize, verbose = 0, aging_active = activation_aging, weights_faults = False,
                                                faulty_addresses = locs, masked_faults = error_mask)
    Accs_M.append(acc)
    fin = time.time()
    time_run = fin - inicio
    run_time_M.append(time_run)
DF_run_time_m = pd.DataFrame(run_time_M)
Acc_MobileNet=pd.DataFrame(Accs_M)

print(str()+' operación completada MobileNet: ', datetime.now().strftime("%H:%M:%S"))

# In[26]:



#Directorio de los pesos
cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'VGG16')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir,'Weights')

Accs_V=[]
run_time_V=[]

trainBatchSize = testBatchSize = 1
_,_,test_dataset = GetDatasets('colorectal_histology',(80,5,15),(224,224), 8, trainBatchSize, testBatchSize)


vol=voltaj[0]

activation_aging = [True] * 21
for i in range(1):
    inicio = time.time()
    # locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/flip_patch/byte_x/vc_707/locs_0' + str(vol))
    # error_mask = load_obj( 'Data/Fault Characterization/variante_mask_vc_707/flip_patch/byte_x/vc_707/error_mask_0' + str(vol))

    vol = vol + paso


    loss,acc   = CheckAccuracyAndLoss('VGG16', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
                                                act_frac_size = 12, act_int_size = 3, wgt_frac_size = 15, wgt_int_size = 0,
                                                batch_size=testBatchSize, verbose = 0, aging_active = activation_aging, weights_faults = False,
                                                faulty_addresses = locs, masked_faults = error_mask)

    Accs_V.append(acc)
    fin = time.time()
    time_run = fin - inicio
    run_time_V.append(time_run)
DF_run_time_v = pd.DataFrame(run_time_V)
Acc_VGG16=pd.DataFrame(Accs_V)
print(acc)
#

#
# # In[30]:
#
print(str()+' operación completada VGG16: ', datetime.now().strftime("%H:%M:%S"))
#
#
# # Directorio de los pesos
cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'ZFNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir,'Weights')

Accs_Z=[]
run_time_Z=[]

trainBatchSize = testBatchSize = 1
_,_,test_dataset = GetDatasets('colorectal_histology',(80,5,15),(224,224), 8, trainBatchSize, testBatchSize)

print('tamaño dle dataset',len(test_dataset))
vol=voltaj[0]
activation_aging = [True] * 11

for i in range(1):
    inicio = time.time()
    # locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/flip_patch/byte_x/vc_707/locs_0' + str(vol))
    # error_mask = load_obj(  'Data/Fault Characterization/variante_mask_vc_707/flip_patch/byte_x/vc_707/error_mask_0' + str(vol))

    vol = vol + paso



    loss,acc   = CheckAccuracyAndLoss('ZFNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
                                                act_frac_size = 11, act_int_size = 4, wgt_frac_size = 15, wgt_int_size = 0,
                                                batch_size=testBatchSize, verbose = 0, aging_active = activation_aging, weights_faults = False,
                                                faulty_addresses = locs, masked_faults = error_mask)

    Accs_Z.append(acc)
    fin = time.time()
    time_run = fin - inicio
    run_time_Z.append(time_run)
DF_run_time_z = pd.DataFrame(run_time_Z)
Acc_ZFNet=pd.DataFrame(Accs_Z)
#
#
# Voltajes=pd.DataFrame(Df_Vol)
# # buf_cero = pd.concat([Voltajes,Acc_AlexNet,Acc_DenseNet,Acc_MobileNet,Acc_SqueezeNet,Acc_VGG16,Acc_ZFNet,Acc_PilotNet], axis=1, join='outer')
# #
# # buf_cero.columns =['Voltajes','AlexNet', 'DenseNet', 'MobileNet', 'SqueezeNet', 'VGG16', 'ZFNet' ,'PilotNet']
# # print(buf_cero)
# # buf_cero.to_excel('F+P_new_acc_error_mask_Base.xlsx', sheet_name='fichero_707', index=False)
#
#
#
# print(Voltajes)
# buf_time = pd.concat([Voltajes,DF_run_time_a, DF_run_time_d,DF_run_time_m ,DF_run_time_s,DF_run_time_v,DF_run_time_z,DF_run_time_p,], axis=1, join='outer')
# buf_time.columns =['Voltajes','AlexNet','DenseNet', 'MobileNet', 'SqueezeNet', 'VGG16', 'ZFNet' ,'PilotNet']
# print(buf_time)
# buf_time.to_excel('Time_error_mask_F_Patch.xlsx', sheet_name='fichero_707', index=False)

buf_time = pd.concat([DF_run_time_a, DF_run_time_d,DF_run_time_m ,DF_run_time_s,DF_run_time_v,DF_run_time_z], axis=1, join='outer')
buf_time.columns =['AlexNet','DenseNet', 'MobileNet', 'SqueezeNet', 'VGG16', 'ZFNet' ]
print(buf_time)
buf_time.to_excel('MoRS/Modelo3_mas_fallos_col_16_sinx_delante/estatics/Time_error_mask_base_Mors.xlsx', sheet_name='fichero_707', index=False)

print(str()+' operación completada: ', datetime.now().strftime("%H:%M:%S"))





#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os
import pickle as pickle
import tensorflow as tf
import numpy as np
from Stats import WeightQuantization,IntroduceFaultsInWeights,GenerateFaultsList,CheckAccuracyAndLoss
from funciones import compilNet, same_elements,buffer_vectores,Base,IsoAECC,ECC, Flip,FlipPatch,FlipPatchBetter,ShiftMask
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



# error_mask= load_obj(     'MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/error_mask_0' )
# locs = load_obj(    'MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/locs_0')
#
# error_mask = load_obj(  'Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/error_mask_054')
# locs = load_obj(  'Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/locs_054')


cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'AlexNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir,'Weights')


run_time_A=[]
change_words=[]
Original_Acc = [0.890666663646697,0.913333356380462,0.881333351135253, 0.93066668510437, 0.805333316326141, 0.833333313465118]
Original_Acc=pd.DataFrame(Original_Acc)


Accs_A=[]
trainBatchSize = testBatchSize = 1
_,_,test_dataset = GetDatasets('colorectal_histology',(80,5,15),(227,227), 8, trainBatchSize, testBatchSize)

#error_mask, locs = Flip(error_mask, locs)


# posiciones=[1,2,3]
# Funcion =[Base,IsoAECC, ECC, Flip,FlipPatch,FlipPatchBetter]
# activation_aging = [True]*11
# for i in range(1):
# #for i in range(len(posiciones)):
# #for i in range(len(Funcion)):
#     inicio = time.time()
#     error_mask= load_obj(     'MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/error_mask_0' )
#     locs = load_obj(    'MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/locs_0')
#
#     #error_mask_new, locs,word_change = Funcion[i](error_mask, locs)
#     error_mask_new, locs,word_change = FlipPatch(error_mask, locs)
#     error_mask_shift,words_fallos= ShiftMask(error_mask_new,posiciones[i])
#     #WordType(error_mask_patch)
#
#
#     print(i)
#     #print('voltaje',vol)
#     print('tamaño de locs',len(locs))
#
#
#     loss,acc   = CheckAccuracyAndLoss('AlexNet', test_dataset, wgt_dir, output_shape=8, input_shape = (227,227,3),
#                                                 act_frac_size = 11, act_int_size = 4, wgt_frac_size = 11, wgt_int_size = 4,
#                                                 batch_size=testBatchSize, verbose = 0, aging_active = activation_aging, weights_faults = False,
#                                                 faulty_addresses = locs, masked_faults = error_mask_shift)
#
#     Accs_A.append(acc)
#     #print(Funcion[i],acc)
#     change_words.append(word_change)
#     fin = time.time()
#     time_run = fin - inicio
#     run_time_A.append(time_run)
# Acc_AlexNet=pd.DataFrame(Accs_A)
# Df_change_words= pd.DataFrame(change_words)
# DF_run_time_a=pd.DataFrame(run_time_A)
# print('Alexnet',acc)
#
#
# print(str()+' operación completada AlexNet: ', datetime.now().strftime("%H:%M:%S"))
#
#
# breakpoint()


#Directorio de los pesos
cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'SqueezeNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir,'Weights')

Accs_S=[]
run_time_S=[]
list_words_fallos_S=[]




trainBatchSize = testBatchSize = 1
_,_,test_dataset = GetDatasets('colorectal_histology',(80,5,15),(224,224), 8, trainBatchSize, testBatchSize)


activation_aging = [True] * 22
#posiciones= [1,2,3]
#for i in range(1):
for i in range(1):
#for i in range(len(posiciones)):
#for i in range(len(Funcion)):
    inicio = time.time()
    error_mask = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/error_mask_0')
    locs = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/locs_0')

    # error_mask_new, locs,word_change = Funcion[i](error_mask, locs)
    error_mask_new, locs,word_change = FlipPatch(error_mask, locs)
    error_mask_shift,words_fallos= ShiftMask(error_mask_new,1)
    print('tamaño de locs', len(locs))
    print('tamaño de error_mask', len(error_mask_new))



    loss,acc   = CheckAccuracyAndLoss('SqueezeNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
                                                act_frac_size = 9, act_int_size = 6, wgt_frac_size = 15, wgt_int_size = 0,
                                                batch_size=testBatchSize, verbose = 0, aging_active = activation_aging, weights_faults = False,
                                                faulty_addresses = locs, masked_faults = error_mask_shift)

    Accs_S.append(acc)
    print(acc)
    fin = time.time()
    time_run = fin - inicio
    run_time_S.append(time_run)
    #list_words_fallos_S.append(words_fallos)
DF_run_time_s = pd.DataFrame(run_time_S)
Acc_SqueezeNet=pd.DataFrame(Accs_S)
#DF_words_fallos_S= pd.DataFrame(list_words_fallos_S)
#print(Acc_SqueezeNet,DF_words_fallos_S)

print('sq', acc)



print(str()+' operación completada SqueezeNet: ', datetime.now().strftime("%H:%M:%S"))




# Directorio de los pesos

# cwd = os.getcwd()
# wgt_dir = os.path.join(cwd, 'Data')
# wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
# wgt_dir = os.path.join(wgt_dir, 'DenseNet')
# wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
# wgt_dir = os.path.join(wgt_dir,'Weights')
#
# Accs_D=[]
# run_time_D=[]
#
# trainBatchSize = testBatchSize = 1
# _,_,test_dataset = GetDatasets('colorectal_histology',(80,5,15),(224,224), 8, trainBatchSize, testBatchSize)
#
#
#
# activation_aging = [True] * 188
# for i in range(1):
# #for i in range(len(posiciones)):
# #for i in range(len(Funcion)):
#     inicio = time.time()
#     error_mask = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/error_mask_0')
#     locs = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/locs_0')
#
#     # error_mask_new, locs,word_change = Funcion[i](error_mask, locs)
#     error_mask_new, locs,word_change = FlipPatch(error_mask, locs)
#     #error_mask_shift,words_fallos= ShiftMask(error_mask_new,posiciones[i])
#
#     loss,acc   = CheckAccuracyAndLoss('DenseNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
#                                                 act_frac_size = 12, act_int_size = 3, wgt_frac_size = 13, wgt_int_size = 2,
#                                                 batch_size=testBatchSize, verbose = 0, aging_active = activation_aging, weights_faults = False,
#                                                 faulty_addresses = locs, masked_faults = error_mask_new)
#
#     Accs_D.append(acc)
#     fin = time.time()
#     time_run = fin - inicio
#     run_time_D.append(time_run)
# DF_run_time_d = pd.DataFrame(run_time_D)
# Acc_DenseNet=pd.DataFrame(Accs_D)
#
#
# # In[22]:
#
# print(str()+' operación completada DenseNet: ', datetime.now().strftime("%H:%M:%S"))
#
#
#
#
# # Directorio de los pesos
# cwd = os.getcwd()
# wgt_dir = os.path.join(cwd, 'Data')
# wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
# wgt_dir = os.path.join(wgt_dir, 'MobileNet')
# wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
# wgt_dir = os.path.join(wgt_dir,'Weights')
# Accs_M=[]
# run_time_M=[]
#
#
# trainBatchSize = testBatchSize = 1
# _,_,test_dataset = GetDatasets('colorectal_histology',(80,5,15),(224,224), 8, trainBatchSize, testBatchSize)
#
#
#
#
# activation_aging = [True]*29
# for i in range(1):
# #for i in range(len(posiciones)):
# #for i in range(len(Funcion)):
#     inicio = time.time()
#     error_mask = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/error_mask_0')
#     locs = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/locs_0')
#
#     # error_mask_new, locs,word_change = Funcion[i](error_mask, locs)
#     error_mask_new, locs,word_change = FlipPatch(error_mask, locs)
#     #error_mask_shift,words_fallos= ShiftMask(error_mask_new,posiciones[i])
#
#
#     loss,acc   = CheckAccuracyAndLoss('MobileNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
#                                                 act_frac_size = 11, act_int_size = 4, wgt_frac_size = 14, wgt_int_size = 1,
#                                                 batch_size=testBatchSize, verbose = 0, aging_active = activation_aging, weights_faults = False,
#                                                 faulty_addresses = locs, masked_faults = error_mask_new)
#     Accs_M.append(acc)
#     fin = time.time()
#     time_run = fin - inicio
#     run_time_M.append(time_run)
# DF_run_time_m = pd.DataFrame(run_time_M)
# Acc_MobileNet=pd.DataFrame(Accs_M)
#
# print(str()+' operación completada MobileNet: ', datetime.now().strftime("%H:%M:%S"))
#
# # In[26]:
#
# #
#
#Directorio de los pesos
# cwd = os.getcwd()
# wgt_dir = os.path.join(cwd, 'Data')
# wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
# wgt_dir = os.path.join(wgt_dir, 'VGG16')
# wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
# wgt_dir = os.path.join(wgt_dir,'Weights')
#
# Accs_V=[]
# run_time_V=[]
# list_words_fallos_V=[]
#
# trainBatchSize = testBatchSize = 1
# _,_,test_dataset = GetDatasets('colorectal_histology',(80,5,15),(224,224), 8, trainBatchSize, testBatchSize)
#
#
#
#
# activation_aging = [True] * 21
# for i in range(1):
# #for i in range(len(posiciones)):
# #for i in range(len(Funcion)):
#     inicio = time.time()
#
#     error_mask = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/error_mask_0')
#     locs = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/locs_0')
#     #error_mask_new, locs,word_change = Funcion[i](error_mask, locs)
#     error_mask_new, locs, word_change = FlipPatch(error_mask, locs)
#     error_mask_shift,words_fallos= ShiftMask(error_mask_new,1)
#     print('tamaño de locs', len(locs))
#     print('tamaño de error_mask', len(error_mask_new))
#
#
#
#     loss,acc   = CheckAccuracyAndLoss('VGG16', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
#                                                 act_frac_size = 12, act_int_size = 3, wgt_frac_size = 15, wgt_int_size = 0,
#                                                 batch_size=testBatchSize, verbose = 0, aging_active = activation_aging, weights_faults = False,
#                                                 faulty_addresses = locs, masked_faults = error_mask_shift)
#
#     Accs_V.append(acc)
#     #list_words_fallos_V.append(words_fallos)
#     fin = time.time()
#     time_run = fin - inicio
#     run_time_V.append(time_run)
# DF_run_time_v = pd.DataFrame(run_time_V)
# Acc_VGG16=pd.DataFrame(Accs_V)
# #DF_words_fallos_V=pd.DataFrame(list_words_fallos_V)
# print('VGG16', acc)
# #print('DF_words_fallos_V', DF_words_fallos_V)
#
#
# # #
# # # # In[30]:
# # #
# print(str()+' operación completada VGG16: ', datetime.now().strftime("%H:%M:%S"))
# #
# # #
# # # # Directorio de los pesos
# cwd = os.getcwd()
# wgt_dir = os.path.join(cwd, 'Data')
# wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
# wgt_dir = os.path.join(wgt_dir, 'ZFNet')
# wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
# wgt_dir = os.path.join(wgt_dir,'Weights')
#
# Accs_Z=[]
# run_time_Z=[]
#
# trainBatchSize = testBatchSize = 1
# _,_,test_dataset = GetDatasets('colorectal_histology',(80,5,15),(224,224), 8, trainBatchSize, testBatchSize)
#
#
#
# activation_aging = [True] * 11
# for i in range(1):
# #for i in range(len(posiciones)):
# #for i in range(len(Funcion)):
#     inicio = time.time()
#
#     error_mask = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/error_mask_0')
#     locs = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/locs_0')
#     #error_mask_new, locs,word_change = Funcion[i](error_mask, locs)
#     error_mask_new, locs, word_change = FlipPatch(error_mask, locs)
#     error_mask_shift,words_fallos= ShiftMask(error_mask_new,1)
#     print('tamaño de locs', len(locs))
#     print('tamaño de error_mask', len(error_mask_new))
#
#
#
#     loss,acc   = CheckAccuracyAndLoss('ZFNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
#                                                 act_frac_size = 11, act_int_size = 4, wgt_frac_size = 15, wgt_int_size = 0,
#                                                 batch_size=testBatchSize, verbose = 0, aging_active = activation_aging, weights_faults = False,
#                                                 faulty_addresses = locs, masked_faults = error_mask_shift)
#
#     Accs_Z.append(acc)
#     fin = time.time()
#     time_run = fin - inicio
#     run_time_Z.append(time_run)
# DF_run_time_z = pd.DataFrame(run_time_Z)
# Acc_ZFNet=pd.DataFrame(Accs_Z)
# #
#
# #
# #
# #
# # #DF_Funcion=pd.DataFrame(['Base','IsoAECC', 'ECC', 'Flip','FlipPatch'])
# DF_Funcion=pd.DataFrame(['Shift 1','Shift 2','Shift 3'])
# # # print('Df_change_words',Df_change_words)
# # # print('DF_Funcion',DF_Funcion)
# # # print('Acc_AlexNet',Acc_AlexNet)
#
# # Shift= pd.concat([DF_Funcion,DF_words_fallos_S,Acc_SqueezeNet,DF_words_fallos_V,Acc_VGG16],axis=1, join='outer')
# # Shift.columns = ['Técnica','Word_fallos', 'Acc_SqueezeNet', 'Word_fallos', 'Acc_VGG16']
# # print(Shift)
# # Shift.to_excel('MoRS/Modelo3_mas_fallos_col_8_experimentos/estatics/Shift_new_squez_vgg16.xlsx', sheet_name='Shift', index=False)
# #
# #
# Shift= pd.concat([DF_Funcion,Acc_AlexNet,Acc_DenseNet,Acc_MobileNet,Acc_SqueezeNet,Acc_VGG16,Acc_ZFNet],axis=1, join='outer')
# Shift.columns = ['Técnica','AlexNet', 'DenseNet', 'MobileNet', 'SqueezeNet', 'VGG16', 'ZFNet']
# print(Shift)
# Shift.to_excel('MoRS/Modelo3_mas_fallos_col_8_experimentos/estatics/inferencia_with_signo_shift_ACC.xlsx', sheet_name='fichero_707', index=False)
# #
# # #
# # # buf_time = pd.concat([DF_run_time_a, DF_run_time_d,DF_run_time_m ,DF_run_time_s,DF_run_time_v,DF_run_time_z], axis=1, join='outer')
# # # buf_time.columns =['AlexNet','DenseNet', 'MobileNet', 'SqueezeNet', 'VGG16', 'ZFNet' ]
# # # print(buf_time)
# # # buf_time.to_excel('MoRS/Modelo3_mas_fallos_col_8_experimentos/estatics/Time_error_mask_flip_Mors.xlsx', sheet_name='Time', index=False)
# # # #
# # # # print(str()+' operación completada: ', datetime.now().strftime("%H:%M:%S"))
# # # #
# # # # #
# # #
# #

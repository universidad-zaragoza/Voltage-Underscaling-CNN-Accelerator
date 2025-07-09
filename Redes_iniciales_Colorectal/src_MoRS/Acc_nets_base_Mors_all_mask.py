#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os
import pickle as pickle
import tensorflow as tf
import numpy as np
from Stats import WeightQuantization, IntroduceFaultsInWeights, GenerateFaultsList, CheckAccuracyAndLoss
from funciones import compilNet, same_elements, buffer_vectores, Base, IsoAECC, ECC, Flip, FlipPatchBetter, ScratchPad,L0flippedHO,Shift, ShiftMask, WordType, VeryBadWords, DeleteTercioRamdom
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
# Acc_net: Desde una ruta definida con los ficgeros de error_mask y locs ya guardados con anterioridad para los voltajes analizados
# los recorre y calcula el acc por cada voltaje para cada red de las estudiadas, finalmente los huarda en un documento excel para un
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


# WordType(error_mask, locs)
# print('tamaño mascar antes', len(error_mask))
# print('tamaño locs antes', len(locs))

####Rectificar siempre el valor de la variable weights_faults en caso de si s equiere inyectar fallos en los pesos o no

# error_mask_H_L_O, locs_H_L_O, index_locs_VBW = VeryBadWords(error_mask, locs)
# error_mask_less_VBW, locs_less_VBW = DeleteTercioRamdom(error_mask, locs, index_locs_VBW)


#voltajes = [0.51, 0.52,0.53, 0.54,0.55, 0.56, 0.57, 0.58]
#voltajes = [0.51, 0.52,0.53]
voltajes = ['0.60']
original = [0.890666663646697, 0.913333356380462, 0.881333351135253,0.93066668510437, 0.805333316326141,0.833333313465118]
vol=voltajes[0]
print('vol',vol)
Funcion = [L0flippedHO]
DF_Funcion = pd.DataFrame(['L0flippedHO'])
#Funcion =[LowOrder,VBWGoToScratch,MaskVeryBadWords,HighOrder]
#Funcion =[HighOrder]
#Tecnicas=['LO','VBWGoToScratch','L&HO','HO']

DF_Funcion.to_excel('MoRS/Modelo3_col_8_'+ str(vol)+'/Analisis_Resultados/estatics/L_F(HO)/test_direccion.xlsx', sheet_name='fichero_707', index=False)




for j , vol  in enumerate(voltajes):
    print('vol', vol)
    inc = 0
    for i in range(2):
        print('Modelo', inc)

        from Stats import CheckAccuracyAndLoss

        error_mask = load_obj('MoRS/Modelo3_col_8_' + str(vol) + '/mask/error_mask_' + str(inc))
        locs = load_obj('MoRS/Modelo3_col_8_' + str(vol) + '/mask/locs_' + str(inc))
        # error_mask = load_obj('Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/error_mask_0' + str(vol))
        # locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/locs_0' + str(vol))
        # print('tamaño mascar adesdues',len(error_mask_new))
        # print('tamaño  locs', len(locs))
        #
        # # Funcion = [Base, IsoAECC, ECC, Flip, FlipPatchBetter, FlipPatch,Shift]

        # Funcion = [ECC]
        run_time_A = []
        change_words = []

        #Original_Acc = pd.DataFrame(Original_Acc)
        Accs_A = []
        #
        cwd = os.getcwd()
        wgt_dir = os.path.join(cwd, 'Data')
        wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
        wgt_dir = os.path.join(wgt_dir, 'AlexNet')
        wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
        wgt_dir = os.path.join(wgt_dir, 'Weights')

        trainBatchSize = testBatchSize = 1
        _, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (227, 227), 8, trainBatchSize, testBatchSize)

        activation_aging = [True] * 11
        # for i in range(1):

        for i in range(len(Funcion)):

            print('Ciclo interno', i)
            inicio = time.time()

            # WordType(error_mask,locs)

            error_mask_new, locs ,word_change= Funcion[i](error_mask, locs)
            WordType(error_mask_new,locs)
            #error_mask_new, locs,locs_modif,word_change = ECC(error_mask, locs)
            print('analizanndo tipo de palabras')
            # WordType(error_mask_new,locs)

            # print('voltaje',vol)
            print('tamaño de locs', len(locs))
            print('tamaño de error_mask_new', len(error_mask_new))

            if Funcion[i] == Shift:
                from Stats_Shift import CheckAccuracyAndLoss
                print('funcion es shift')

            loss, acc = CheckAccuracyAndLoss('AlexNet', test_dataset, wgt_dir, output_shape=8, input_shape=(227, 227, 3),
                                             act_frac_size=11, act_int_size=4, wgt_frac_size=11, wgt_int_size=4,
                                             batch_size=testBatchSize,
                                             verbose=0, aging_active=activation_aging, weights_faults=False,
                                             faulty_addresses=locs, masked_faults=error_mask_new)

            Accs_A.append(acc)
            print(Funcion[i], acc)
            fin = time.time()
            time_run = fin - inicio
            run_time_A.append(time_run)
        change_words.append(word_change)
        Acc_AlexNet = pd.DataFrame(Accs_A)
        Df_change_words = pd.DataFrame(change_words)
        DF_run_time_a = pd.DataFrame(run_time_A)
        print('Acc_AlexNet', Acc_AlexNet)

        print(str() + ' operación completada AlexNet: ', datetime.now().strftime("%H:%M:%S"))

        # Directorio de los pesos
        cwd = os.getcwd()
        wgt_dir = os.path.join(cwd, 'Data')
        wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
        wgt_dir = os.path.join(wgt_dir, 'SqueezeNet')
        wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
        wgt_dir = os.path.join(wgt_dir, 'Weights')

        Accs_S = []
        run_time_S = []
        list_words_fallos_S = []

        trainBatchSize = testBatchSize = 1
        _, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)

        activation_aging = [True] * 22
        # posiciones= [1,2,3]
        # for i in range(1):
        # for i in range(len(posiciones)):

        for i in range(len(Funcion)):
            from Stats import CheckAccuracyAndLoss

            print('Ciclo interno', i)
            inicio = time.time()

            error_mask_new, locs, word_change = Funcion[i](error_mask, locs)

            # WordType(error_mask_patch)
            # error_mask_new, locs,locs_modif,word_change = ScratchPad(error_mask, locs)
            # error_mask_shift,words_fallos= ShiftMask(error_mask_new,posiciones[i])
            print('tamaño de locs', len(locs))
            print('tamaño de error_mask', len(error_mask_new))
            if Funcion[i] == Shift:
                from Stats_Shift import CheckAccuracyAndLoss
            loss, acc = CheckAccuracyAndLoss('SqueezeNet', test_dataset, wgt_dir, output_shape=8, input_shape=(224, 224, 3),
                                             act_frac_size=9, act_int_size=6, wgt_frac_size=15, wgt_int_size=0,
                                             batch_size=testBatchSize, verbose=0, aging_active=activation_aging,
                                             weights_faults=False, faulty_addresses=locs, masked_faults=error_mask_new)

            Accs_S.append(acc)
            print(acc)
            fin = time.time()
            time_run = fin - inicio
            run_time_S.append(time_run)
            # list_words_fallos_S.append(words_fallos)
        DF_run_time_s = pd.DataFrame(run_time_S)
        Acc_SqueezeNet = pd.DataFrame(Accs_S)
        # DF_words_fallos_S = pd.DataFrame(list_words_fallos_S)
        print('Acc_SqueezeNet', Acc_SqueezeNet)

        print(str() + ' operación completada SqueezeNet: ', datetime.now().strftime("%H:%M:%S"))
        #
        # Directorio de los pesos
        cwd = os.getcwd()
        wgt_dir = os.path.join(cwd, 'Data')
        wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
        wgt_dir = os.path.join(wgt_dir, 'MobileNet')
        wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
        wgt_dir = os.path.join(wgt_dir, 'Weights')
        Accs_M = []
        run_time_M = []

        trainBatchSize = testBatchSize = 1
        _, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)

        activation_aging = [True] * 29

        for i in range(len(Funcion)):
            # for i in range(2):
            print('Ciclo interno', i)
            from Stats import CheckAccuracyAndLoss

            inicio = time.time()

            error_mask_new, locs,  word_change = Funcion[i](error_mask, locs)
            # WordType(error_mask_patch)
            print('tamaño de locs', len(locs))
            print('tamaño de error_mask', len(error_mask_new))
            if Funcion[i] == Shift:
                from Stats_Shift import CheckAccuracyAndLoss
            loss, acc = CheckAccuracyAndLoss('MobileNet', test_dataset, wgt_dir, output_shape=8, input_shape=(224, 224, 3),
                                             act_frac_size=11, act_int_size=4, wgt_frac_size=14, wgt_int_size=1,
                                             batch_size=testBatchSize, verbose=0, aging_active=activation_aging,
                                             weights_faults=False, faulty_addresses=locs, masked_faults=error_mask_new)
            Accs_M.append(acc)
            fin = time.time()
            time_run = fin - inicio
            run_time_M.append(time_run)
        DF_run_time_m = pd.DataFrame(run_time_M)
        Acc_MobileNet = pd.DataFrame(Accs_M)
        print('Acc_MobileNet', Acc_MobileNet)

        print(str() + ' operación completada MobileNet: ', datetime.now().strftime("%H:%M:%S"))

        # Directorio de los pesos
        cwd = os.getcwd()
        wgt_dir = os.path.join(cwd, 'Data')
        wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
        wgt_dir = os.path.join(wgt_dir, 'VGG16')
        wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
        wgt_dir = os.path.join(wgt_dir, 'Weights')

        Accs_V = []
        run_time_V = []
        list_words_fallos_V = []

        trainBatchSize = testBatchSize = 1
        _, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)

        activation_aging = [True] * 21
        # for i in range(1):
        for i in range(len(Funcion)):
            print('Ciclo interno', i)
            from Stats import CheckAccuracyAndLoss

            inicio = time.time()

            error_mask_new, locs,  word_change = Funcion[i](error_mask, locs)
            # WordType(error_mask_patch)
            # error_mask_new, locs,locs_modif,word_change = ScratchPad(error_mask, locs)
            # error_mask_shift,words_fallos= ShiftMask(error_mask_new,posiciones[i])
            print('tamaño de locs', len(locs))
            print('tamaño de error_mask', len(error_mask_new))
            if Funcion[i] == Shift:
                from Stats_Shift import CheckAccuracyAndLoss
            loss, acc = CheckAccuracyAndLoss('VGG16', test_dataset, wgt_dir, output_shape=8, input_shape=(224, 224, 3),
                                             act_frac_size=12, act_int_size=3, wgt_frac_size=15, wgt_int_size=0,
                                             batch_size=testBatchSize, verbose=0, aging_active=activation_aging,
                                             weights_faults=False,faulty_addresses=locs, masked_faults=error_mask_new)

            Accs_V.append(acc)
            # list_words_fallos_V.append(words_fallos)
            fin = time.time()
            time_run = fin - inicio
            run_time_V.append(time_run)
        DF_run_time_v = pd.DataFrame(run_time_V)
        Acc_VGG16 = pd.DataFrame(Accs_V)
        # DF_words_fallos_V = pd.DataFrame(list_words_fallos_V)
        print('Acc_VGG16', Acc_VGG16)

        print(str() + ' operación completada VGG16: ', datetime.now().strftime("%H:%M:%S"))

        #  Directorio de los pesos
        cwd = os.getcwd()
        wgt_dir = os.path.join(cwd, 'Data')
        wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
        wgt_dir = os.path.join(wgt_dir, 'ZFNet')
        wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
        wgt_dir = os.path.join(wgt_dir, 'Weights')

        Accs_Z = []
        run_time_Z = []

        trainBatchSize = testBatchSize = 1
        _, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)

        activation_aging = [True] * 11
        # for i in range(1):

        for i in range(len(Funcion)):

            print('Ciclo interno', i)
            from Stats import CheckAccuracyAndLoss

            inicio = time.time()

            error_mask_new, locs,  word_change = Funcion[i](error_mask, locs)
            # WordType(error_mask_patch)
            print('tamaño de locs', len(locs))
            print('tamaño de error_mask', len(error_mask_new))
            if Funcion[i] == Shift:
                from Stats_Shift import CheckAccuracyAndLoss
            loss, acc = CheckAccuracyAndLoss('ZFNet', test_dataset, wgt_dir, output_shape=8, input_shape=(224, 224, 3),
                                             act_frac_size=11, act_int_size=4, wgt_frac_size=15, wgt_int_size=0,
                                             batch_size=testBatchSize, verbose=0, aging_active=activation_aging,
                                             weights_faults=False,faulty_addresses=locs, masked_faults=error_mask_new)

            Accs_Z.append(acc)
            fin = time.time()
            time_run = fin - inicio
            run_time_Z.append(time_run)
        DF_run_time_z = pd.DataFrame(run_time_Z)
        Acc_ZFNet = pd.DataFrame(Accs_Z)
        print('Acc_ZFNet', Acc_ZFNet)

        # Directorio de los pesos

        cwd = os.getcwd()
        wgt_dir = os.path.join(cwd, 'Data')
        wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
        wgt_dir = os.path.join(wgt_dir, 'DenseNet')
        wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
        wgt_dir = os.path.join(wgt_dir, 'Weights')

        Accs_D = []
        run_time_D = []

        trainBatchSize = testBatchSize = 1
        _, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)
        #
        activation_aging = [True] * 188
        # for i in range(1):

        for i in range(len(Funcion)):
            print('Ciclo interno', i)
            from Stats import CheckAccuracyAndLoss

            inicio = time.time()

            error_mask_new, locs,  word_change = Funcion[i](error_mask, locs)
            # WordType(error_mask_patch)
            print('tamaño de locs', len(locs))
            print('tamaño de error_mask', len(error_mask_new))
            if Funcion[i] == Shift:
                from Stats_Shift import CheckAccuracyAndLoss
            loss, acc = CheckAccuracyAndLoss('DenseNet', test_dataset, wgt_dir, output_shape=8, input_shape=(224, 224, 3),
                                             act_frac_size=12, act_int_size=3, wgt_frac_size=13, wgt_int_size=2,
                                             batch_size=testBatchSize, verbose=0, aging_active=activation_aging,
                                             weights_faults=False, faulty_addresses=locs, masked_faults=error_mask_new)

            Accs_D.append(acc)
            fin = time.time()
            time_run = fin - inicio
            run_time_D.append(time_run)
        DF_run_time_d = pd.DataFrame(run_time_D)
        Acc_DenseNet = pd.DataFrame(Accs_D)
        print('Acc_DenseNet', Acc_DenseNet)

        # Funcion =[Base,IsoAECC, ECC, Flip,FlipPatch,FlipPatch]
        #DF_Funcion = pd.DataFrame(['Base', 'ECC'])
        #print('Df_change_words', Df_change_words)
        #     print('DF_Funcion',DF_Funcion)

        # Shift= pd.concat([DF_Funcion,DF_words_fallos_S,Acc_SqueezeNet,DF_words_fallos_V,Acc_VGG16],axis=1, join='outer')
        # Shift.columns = ['Técnica','Word_fallos', 'Acc_SqueezeNet', 'Word_fallos', 'Acc_VGG16']
        # print(Shift)
        # Shift.to_excel('MoRS/Modelo3_mas_fallos_col_8_experimentos/estatics/Shift_squez_vgg16.xlsx', sheet_name='Shift', index=False)
        # #

        print(str() + ' operación completada DenseNet: ', datetime.now().strftime("%H:%M:%S"))

        Acc_all_exp = pd.concat(   [Df_change_words, DF_Funcion, Acc_AlexNet, Acc_DenseNet, Acc_MobileNet, Acc_SqueezeNet, Acc_VGG16, Acc_ZFNet],
            axis=1, join='outer')
        print(Acc_all_exp)
        Acc_all_exp.columns = ['words_changej', 'Técnica', 'AlexNet', 'DenseNet', 'MobileNet', 'SqueezeNet', 'VGG16',
                               'ZFNet']
        Acc_all_exp = pd.concat([pd.DataFrame([None] * Acc_all_exp.shape[1], index=Acc_all_exp.columns).T, Acc_all_exp],
                                ignore_index=True)
        Acc_all_exp.loc[0] = [''] + ['original'] + original
        print(Acc_all_exp)
        Acc_all_exp.to_excel( 'MoRS/Modelo3_col_8_' + str(vol) + '/Analisis_Resultados/estatics/L_F(HO)/ACC_all_Experiment_modelo_' + str(
                vol) + '_' + str(inc) + '.xlsx', sheet_name='fichero_707', index=False)

        # Acc_all_exp = pd.concat([Df_change_words, DF_Funcion, Acc_AlexNet],
        #     axis=1, join='outer')
        # Acc_all_exp.columns = ['words_change', 'Técnica', 'AlexNet']
        # print(Acc_all_exp)
        #
        # buf_time = pd.concat([DF_run_time_a, DF_run_time_d, DF_run_time_m, DF_run_time_s, DF_run_time_v, DF_run_time_z],
        #                      axis=1, join='outer')
        # buf_time.columns = ['AlexNet', 'DenseNet', 'MobileNet', 'SqueezeNet', 'VGG16', 'ZFNet']
        # print(buf_time)
        # Acc_all_exp.to_excel( 'MoRS/Modelo3_col_8_' + str(vol) + '/Analisis_Resultados/estatics/L_F(HO)/ACC_all_Experiment_modelo_test_' + str(vol) + '_' + str(inc) + '.xlsx', sheet_name='fichero_707', index=False)
        # #
        inc = inc + 1



print(str()+' operación completada: ', datetime.now().strftime("%H:%M:%S"))
#
# # #
#
#

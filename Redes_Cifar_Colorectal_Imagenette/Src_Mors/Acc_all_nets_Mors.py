#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os
import pickle as pickle
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.optimizers import Adam
import numpy as np
from Stats import WeightQuantization, IntroduceFaultsInWeights, GenerateFaultsList, CheckAccuracyAndLoss
from funciones import compilNet, same_elements,buffer_vectores,Base,IsoAECC,ECC, Flip,FlipPatchBetter,ScratchPad,Shift,L0flippedHO
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


# test_save = pd.DataFrame(['Base', 'ECC', 'Flip', 'FlipPatch', 'ScratchPad', 'Shift2'])
# test_save.columns = ['DF_Funcion']
# print(test_save)
# test_save.to_excel('MoRS/Modelo3_col_8_' + str(vol) + '/Analisis_Resultados/estatics/ACC_all_Experiment_modelo_' + str(vol) + '_'+ str(inc) + '.xlsx', sheet_name='Time', index=False)




# error_mask = load_obj('Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/error_mask_054')
# locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/locs_054')
#
# error_mask = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/error_mask_0')
# locs = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/locs_0')
#voltajes = [0.51,0.52, 0.53, 0.54,0.55, 0.56, 0.57, 0.58,0.59]
#voltajes = [0.51, 0.52,0.53]
voltajes = ['0.53']
original = [ 0.768000007, 0.8119999, 0.944000005722045,0.899999976,]
vol=voltajes[0]
print('vol',vol)
DF_Funcion = pd.DataFrame(['L0flippedHO'])
DF_Funcion.to_excel('MoRS/Modelo3_col_8_'+ str(vol)+'/Analisis_Resultados/estatics/L_F(HO)/test_direccion.xlsx', sheet_name='fichero_707', index=False)

#vol = 54
for j , vol  in enumerate(voltajes):
    print('vol', vol)
    inc = 0

    for j in range(2):
        print('inc', inc)
        print('Ciclo externo',j)
        from Stats import CheckAccuracyAndLoss

        Funcion = [L0flippedHO]
        DF_Funcion = pd.DataFrame(['L0flippedHO'])
        # Funcion =[LowOrder,VBWGoToScratch,MaskVeryBadWords,HighOrder]
        # Funcion =[HighOrder]
        # Tecnicas=['LO','VBWGoToScratch','L&HO','HO']

        #Funcion = [Base, ECC, Flip, FlipPatchBetter, FlipPatch, Shift]
        #Funcion = [Base, ECC]
        # error_mask = load_obj('home/yamilka/cnn_gating/MoRS/Modelo3_col_8_' + str(vol) + '/mask/error_mask_' + str(inc))
        # locs = load_obj('home/yamilka/cnn_gating/MoRS/Modelo3_col_8_' + str(vol) + '/mask/locs_' + str(inc))

        error_mask = load_obj('MoRS/Modelo3_col_8_' + str(vol) + '/mask/error_mask_' + str(inc))
        locs = load_obj('MoRS/Modelo3_col_8_' + str(vol) + '/mask/locs_' + str(inc))
    #
        wgt_dir= ('../weights/Xception/weights.data')
    #
    # #
    # #
        (train_ds, validation_ds, test_ds), info = tfds.load(    "colorectal_histology",    split=["train[:85%]", "train[85%:95%]", "train[95%:]"],
            with_info=True,    as_supervised=True,    shuffle_files= True
        )

        num_classes = info.features['label'].num_classes

        size = (150, 150)
        batch_size = 1

        train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
        validation_ds = validation_ds.map(lambda x, y: (tf.image.resize(x, size), y))
        test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, size), y))

        train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=1)
        validation_ds = validation_ds.cache().batch(batch_size).prefetch(buffer_size=1)
        test_ds = test_ds.cache().batch(batch_size).prefetch(buffer_size=1)

    #
    #
        run_time_A=[]
        #

        Redes=[]
        All_acc =[]
        All_acc_normal = []



        run_time_X=[]
        Accs_X=[]
        list_words_fallos_X=[]
        activation_aging = [True]*47

        #posiciones= [1,2,3]
        #for i in range(len(posiciones)):
        #for i in range(1):
        for i in range(len(Funcion)):
            from Stats import CheckAccuracyAndLoss
            inicio = time.time()
            #print('tectica',Funcion[i])


            error_mask_new, locs ,word_change= Funcion[i](error_mask, locs)
            #error_mask_new, locs,word_change = FlipPatch(error_mask, locs)
            #error_mask_shift,words_fallos= ShiftMask(error_mask_new,1)
            print('error_mask_new', len(error_mask_new))
            print('locs', len(locs))

            if Funcion[i]==Shift:
                from Stats_Shift import CheckAccuracyAndLoss
                print('es flip con shift ahora')
            ## con esto corrimos el experimento para acativaciones
            loss, acc = CheckAccuracyAndLoss('Xception', test_ds, wgt_dir, output_shape=8, input_shape=(150, 150, 3),
                                             act_frac_size=7, act_int_size=8, wgt_frac_size=9, wgt_int_size=11,
                                             batch_size=batch_size, verbose=0, aging_active=activation_aging,
                                             weights_faults=False,  faulty_addresses=locs, masked_faults=error_mask_new)



            Accs_X.append(acc)
            print('acc', acc)
            #print('Tecnica', posiciones[i])
            #All_acc.append(acc)
            #acc_normal = acc / 0.903999984
            #All_acc_normal.append(acc_normal)
            fin = time.time()
            time_run = fin - inicio
            run_time_X.append(time_run)
            list_words_fallos_X.append(word_change)



        Acc_Xception=pd.DataFrame(Accs_X)
        print('Acc_Xception',Acc_Xception)
        print('All_acc',acc)
        DF_run_time_X=pd.DataFrame(run_time_X)
        DF_words_fallos_X= pd.DataFrame(list_words_fallos_X)

        print(str()+' operación completada Xception: ', datetime.now().strftime("%H:%M:%S"))



        #
        wgt_dir= ('../weights/Inception/weights.data')



        (train_ds, validation_ds, test_ds), info = tfds.load(
            "colorectal_histology",
            split=["train[:85%]", "train[85%:95%]", "train[95%:]"],
            with_info=True,
            as_supervised=True,
            shuffle_files= True
        )

        num_classes = info.features['label'].num_classes

        size = (150, 150)
        batch_size = 1

        train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
        validation_ds = validation_ds.map(lambda x, y: (tf.image.resize(x, size), y))
        test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, size), y))

        train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=1)
        validation_ds = validation_ds.cache().batch(batch_size).prefetch(buffer_size=1)
        test_ds = test_ds.cache().batch(batch_size).prefetch(buffer_size=1)

        Accs_I=[]
        run_time_I = []
        list_words_fallos_I=[]

        activation_aging = [True]*170

        #posiciones= [1,2,3]
        #for i in range(len(posiciones)):
        #for i in range(1):
        for i in range(len(Funcion)):
            from Stats import CheckAccuracyAndLoss
            inicio = time.time()
            error_mask_new, locs ,word_change= Funcion[i](error_mask, locs)
            #error_mask_new, locs, word_change = FlipPatch(error_mask, locs)
            #error_mask_shift,words_fallos= ShiftMask(error_mask_new,1)
            print('error_mask_new', len(error_mask_new))
            print('locs', len(locs))

            if Funcion[i]==Shift:
                print('es flip con shift ahora')
                from Stats_Shift import CheckAccuracyAndLoss
        #     ## con esto corrimos el experimento para cativaciones
            loss, acc = CheckAccuracyAndLoss('Inception', test_ds, wgt_dir, output_shape=8, input_shape=(150, 150, 3),
                                             act_frac_size=7, act_int_size=8, wgt_frac_size=11, wgt_int_size=12,
                                             batch_size=batch_size, verbose=0, aging_active=activation_aging,
                                             weights_faults=True,faulty_addresses=locs, masked_faults=error_mask_new)

            Accs_I.append(acc)
            print('acc f_P',acc)
            #All_acc.append(acc)
            #acc_normal = acc / 0.768000007
            #All_acc_normal.append(acc_normal)
            fin = time.time()
            time_run = fin - inicio
            run_time_I.append(time_run)
            list_words_fallos_I.append(word_change)


        Acc_Inception=pd.DataFrame(Accs_I)
        print('Acc_Inception',Acc_Inception)
        print('All_acc',acc)
        DF_run_time_I=pd.DataFrame(run_time_I)
        DF_words_fallos_I= pd.DataFrame(list_words_fallos_I)
        #
        print(str()+' operación completada Inception: ', datetime.now().strftime("%H:%M:%S"))


            # Directorio de los pesos
        wgt_dir= ('../weights/VGG19/weights.data')

        run_time_V19=[]
        Accs_V =[]
        activation_aging = [True]*28
        #for i in range(1):
        for i in range(len(Funcion)):
            from Stats import CheckAccuracyAndLoss
            inicio = time.time()
            error_mask_new, locs ,word_change= Funcion[i](error_mask, locs)
            #error_mask_new, locs,word_change = FlipPatch(error_mask, locs)
            #error_mask_shift,words_fallos= ShiftMask(error_mask_new,1)
            print('error_mask_new', len(error_mask_new))
            print('locs', len(locs))


            if Funcion[i]==Shift:
                print('es flip con shift ahora')
                from Stats_Shift import CheckAccuracyAndLoss
            loss, acc = CheckAccuracyAndLoss('VGG19', test_ds, wgt_dir, output_shape=8, input_shape=(150, 150, 3),
                                             act_frac_size=7, act_int_size=8, wgt_frac_size=15, wgt_int_size=0,
                                             batch_size=batch_size, verbose=0, aging_active=activation_aging,
                                             weights_faults=True,   faulty_addresses=locs, masked_faults=error_mask_new)
            Accs_V.append(acc)
            print(acc)
            #All_acc.append(acc)
            #acc_normal = acc / 0.944000005722045
            #All_acc_normal.append(acc_normal)
            fin = time.time()
            time_run = fin - inicio
            run_time_V19.append(time_run)
        Acc_VGG19=pd.DataFrame(Accs_V)
        print('Acc_VGG19',Acc_VGG19)
        print('All_acc',acc)
        DF_run_time_V19=pd.DataFrame(run_time_V19)


        print(str()+' operación VGG19: ', datetime.now().strftime("%H:%M:%S"))
        #
    #     #
        wgt_dir= ('../weights/ResNet50/weights.data')


        (train_ds, validation_ds, test_ds), info = tfds.load(
            "colorectal_histology",
            split=["train[:85%]", "train[85%:95%]", "train[95%:]"],
            with_info=True,
            as_supervised=True,
            shuffle_files=True,
        )

        num_classes = info.features['label'].num_classes

        size = (150, 150)
        batch_size = 1

        train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
        validation_ds = validation_ds.map(lambda x, y: (tf.image.resize(x, size), y))
        test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, size), y))

        train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=1)
        validation_ds = validation_ds.cache().batch(batch_size).prefetch(buffer_size=1)
        test_ds = test_ds.cache().batch(batch_size).prefetch(buffer_size=1)

        run_time_R=[]
        Accs_R= []

        #activation_aging = [True]*22
        activation_aging = [True] * 52

        #for i in range(1):
        for i in range(len(Funcion)):
            from Stats import CheckAccuracyAndLoss

            inicio = time.time()

            error_mask_new, locs ,word_change= Funcion[i](error_mask, locs)
            print('Funcion[i]',Funcion[i])
            #error_mask_shift,words_fallos= ShiftMask(error_mask_new,1)

            print('error_mask_new', len(error_mask_new))
            print('locs', len(locs))


            if Funcion[i]==Shift:
                from Stats_Shift import CheckAccuracyAndLoss
                print('es flip con shift ahora')
            loss, acc = CheckAccuracyAndLoss('ResNet', test_ds, wgt_dir, output_shape=8, input_shape=(150, 150, 3),
                                             act_frac_size=11, act_int_size=4, wgt_frac_size=14, wgt_int_size=1,
                                             batch_size=batch_size, verbose=0, aging_active=activation_aging,
                                             weights_faults=True, faulty_addresses=locs, masked_faults=error_mask_new)

            Accs_R.append(acc)
            print('acc',acc)
            #All_acc.append(acc)
            #acc_normal = acc / 0.8119999
            #All_acc_normal.append(acc_normal)
            fin = time.time()
            time_run = fin - inicio
            run_time_R.append(time_run)
            del error_mask_new
        Acc_ResNet = pd.DataFrame(Accs_R)
        print('Acc_ResNet', Acc_ResNet)

        #DF_Funcion = pd.DataFrame(['Base', 'ECC', 'Flip', 'FlipPatch', 'ScratchPad', 'Shift2'])

        Acc_all_exp = pd.concat( [DF_words_fallos_I, DF_Funcion, Acc_Inception,Acc_ResNet,Acc_VGG19,Acc_Xception,],  axis=1, join='outer')
        Acc_all_exp.columns = ['words_change', 'Técnica', 'Inception','ResNet', 'VGG19','Xception']
        Acc_all_exp = pd.concat([pd.DataFrame([None] * Acc_all_exp.shape[1], index=Acc_all_exp.columns).T, Acc_all_exp],
                                ignore_index=True)
        Acc_all_exp.loc[0] = [''] + ['original'] + original
        print(Acc_all_exp)
        Acc_all_exp.to_excel('MoRS/Modelo3_col_8_' + str(vol) + '/Analisis_Resultados/estatics/L_F(HO)/ACC_all_Experiment_modelo_test' + str(vol) + '_'+ str(inc) + '.xlsx', sheet_name='Time', index=False)



        # DF_Funcion = pd.DataFrame(['Base', 'ECC', 'Flip', 'FlipPatch', 'ScratchPad', 'Shift2'])
        # #DF_Funcion = pd.DataFrame(['Shift2'])
        # Acc_all_exp = pd.concat([ DF_Funcion, Acc_ResNet], axis=1, join='outer')
        # Acc_all_exp.columns = [ 'Técnica',  'ResNet']
        # print(Acc_all_exp)
        # Acc_all_exp = pd.concat([DF_words_fallos_I, DF_Funcion,Acc_Inception, Acc_ResNet,  Acc_VGG19,Acc_Xception], axis=1,
        #                         join='outer')
        # Acc_all_exp.columns = ['words_change', 'Técnica', 'Inception','ResNet',  'VGG19', 'Xception']
        # print(Acc_all_exp)

        # Acc_all_exp.to_excel('MoRS/Modelo3_col_8_' + str(vol) + '/Analisis_Resultados/estatics/ACC_all_Resnet_Experiment_modelo_' + str(vol) + '_' + str(inc) + '.xlsx',
        #                      sheet_name='ResNet', index=False)
        # #

        # buf_time = pd.concat([DF_run_time_R,DF_run_time_X, DF_run_time_I,DF_run_time_V19 ], axis=1, join='outer')
        # buf_time.columns =['ResNet','Xception','Inception', 'VGG19']
        # print(buf_time)
        # buf_time.to_excel('MoRS/Modelo3_mas_fallos_col_8_experimentos/estatics/Time_error_mask_053_'+ str(inc) + '.xlsx', sheet_name='fichero_707', index=False)

        inc = inc + 1
print(str()+' operación completada: ', datetime.now().strftime("%H:%M:%S"))





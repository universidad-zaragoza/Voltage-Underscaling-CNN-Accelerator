#!/usr/bin/env python
# coding: utf-8

# In[15]:

#### Dada un amáscara y los direcciones con fallos , las va transformando en LO. HO y L&HO
## y va calculando el acc par acada red y guardando enlistas, finalmente se crea un fataframe y se guarda
#el resultado obtenido para cada técnica
import os
import pickle as pickle
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.optimizers import Adam
import numpy as np
from Stats import WeightQuantization, IntroduceFaultsInWeights, GenerateFaultsList, CheckAccuracyAndLoss
from funciones import compilNet, same_elements,buffer_vectores,LowOrder,HighOrder,VeryBadWords,VBWGoToScratch,MaskVeryBadWords,DeleteTercioRamdom,L0flippedHO
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

vol=0.51
inc = 0
# Tecnicas=['LO','VBWGoToScratch','L&HO','HO']
# Funcion =[LowOrder,VBWGoToScratch,MaskVeryBadWords,HighOrder]
#Tecnicas=['LO','VBWGoToScratch','L&HO','HO']
Funcion =[L0flippedHO,VBWGoToScratch]
Tecnicas=['L0flippedHO','VBWGoToScratch']
Df_Tecn=pd.DataFrame(Tecnicas*4)
print(Df_Tecn)
Df_Tecn.to_excel('MoRS/Modelo3_col_8_'+ str(vol)+'/Analisis_Resultados/Palabras_x_tipo/Mask_less_LHO_L_F(HO)/test' + str(inc)+'.xlsx', sheet_name='fichero_707', index=False)





# In[ ]:

#voltaj=[54,55,56,57,58,59,60]
#Df_Vol=[54,56,58,60]
#Voltajes=pd.DataFrame(Df_Vol)

for j in range(10):

    error_mask = load_obj('MoRS/Modelo3_col_8_' + str(vol) + '/mask/error_mask_' + str(inc))
    locs = load_obj('MoRS/Modelo3_col_8_' + str(vol) + '/mask/locs_' + str(inc))

    print('tamaño de la mascara inicial', len(error_mask))
    print('tamaño de la locs inicial', len(locs))

    print('Modelo', inc)
    # Este codigo lo uso para voltaje sinferiores a 0.53
    error_mask_H_L_O, locs_H_L_O, index_locs_VBW = VeryBadWords(error_mask, locs)
    error_mask_less_VBW, locs_less_VBW = DeleteTercioRamdom(error_mask, locs, index_locs_VBW)
    print('tamaño de la mascara final', len(error_mask_less_VBW))
    print('tamaño de la locs final', len(locs_less_VBW))



    wgt_dir= ('../weights/Xception/weights.data')

    #
    #
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

    Redes=[]
    All_acc =[]
    All_acc_normal = []
    Accs_X=[]
    activation_aging = [True]*47


    for i in range(len(Funcion)):


        #error_mask_new, locs = Funcion[i](error_mask, locs)
        error_mask_new, locs,word_change = Funcion[i](error_mask_less_VBW, locs_less_VBW)
        print('error_mask_new', len(error_mask_new))
        print('locs', len(locs))


        ## con esto corrimos el experimento para acativaciones
        loss, acc = CheckAccuracyAndLoss('Xception', test_ds, wgt_dir, output_shape=8, input_shape=(150, 150, 3),
                                         act_frac_size=7, act_int_size=8, wgt_frac_size=9, wgt_int_size=11,
                                         batch_size=batch_size, verbose=0, aging_active=activation_aging,
                                         weights_faults=False,  faulty_addresses=locs, masked_faults=error_mask_new)



        Accs_X.append(acc)
        All_acc.append(acc)
        acc_normal = acc / 0.899999976
        All_acc_normal.append(acc_normal)
        del error_mask_new
        Redes.append('Xception')

    Acc_Xception=pd.DataFrame(Accs_X)
    print('Acc_Xception',Acc_Xception)
    print('All_acc',All_acc)

    print(str()+' operación completada Xception: ', datetime.now().strftime("%H:%M:%S"))


    #
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

    activation_aging = [True]*170

    for i in range(len(Funcion)):


        #error_mask_new, locs = Funcion[i](error_mask, locs)
        error_mask_new, locs,word_change = Funcion[i](error_mask_less_VBW, locs_less_VBW)
        print('error_mask_new', len(error_mask_new))
        print('locs', len(locs))


    #     ## con esto corrimos el experimento para cativaciones
        loss, acc = CheckAccuracyAndLoss('Inception', test_ds, wgt_dir, output_shape=8, input_shape=(150, 150, 3),
                                         act_frac_size=7, act_int_size=8, wgt_frac_size=11, wgt_int_size=12,
                                         batch_size=batch_size, verbose=0, aging_active=activation_aging,
                                         weights_faults=False,faulty_addresses=locs, masked_faults=error_mask_new)

        Accs_I.append(acc)
        All_acc.append(acc)
        acc_normal = acc / 0.768000007
        All_acc_normal.append(acc_normal)
        del error_mask_new
        Redes.append('Inception')

    Acc_Inception=pd.DataFrame(Accs_I)
    print('Acc_Inception',Acc_Inception)
    print('All_acc',All_acc)

    print(str()+' operación completada Inception: ', datetime.now().strftime("%H:%M:%S"))


        # Directorio de los pesos
    wgt_dir= ('../weights/VGG19/weights.data')


    Accs_V =[]
    activation_aging = [True]*28

    for i in range(len(Funcion)):

        #error_mask_new, locs = Funcion[i](error_mask, locs)
        error_mask_new, locs,word_change = Funcion[i](error_mask_less_VBW, locs_less_VBW)
        print('error_mask_new', len(error_mask_new))
        print('locs', len(locs))



        loss, acc = CheckAccuracyAndLoss('VGG19', test_ds, wgt_dir, output_shape=8, input_shape=(150, 150, 3),
                                         act_frac_size=7, act_int_size=8, wgt_frac_size=15, wgt_int_size=0,
                                         batch_size=batch_size, verbose=0, aging_active=activation_aging,
                                         weights_faults=False,   faulty_addresses=locs, masked_faults=error_mask_new)
        Accs_V.append(acc)
        print(acc)
        All_acc.append(acc)
        acc_normal = acc / 0.944000005722045
        All_acc_normal.append(acc_normal)
        del error_mask_new
        Redes.append('VGG19')
    Acc_VGG19=pd.DataFrame(Accs_V)
    print('Acc_VGG19',Acc_VGG19)
    print('All_acc',All_acc)
    #

    print(str()+' operación VGG19: ', datetime.now().strftime("%H:%M:%S"))
    #

    wgt_dir= ('../weights/ResNet50/weights.data')



    Accs_R= []

    activation_aging = [True]*22


    for i in range(len(Funcion)):
        print(i)

        #error_mask_new, locs = Funcion[i](error_mask, locs)
        error_mask_new, locs,word_change = Funcion[i](error_mask_less_VBW, locs_less_VBW)

        print('error_mask_new', len(error_mask_new))
        print('locs', len(locs))



        loss, acc = CheckAccuracyAndLoss('ResNet', test_ds, wgt_dir, output_shape=8, input_shape=(150, 150, 3),
                                         act_frac_size=11, act_int_size=4, wgt_frac_size=14, wgt_int_size=1,
                                         batch_size=batch_size, verbose=0, aging_active=activation_aging,
                                         weights_faults=False, faulty_addresses=locs, masked_faults=error_mask_new)

        Accs_R.append(acc)
        print('acc',acc)
        All_acc.append(acc)
        acc_normal = acc / 0.8119999
        All_acc_normal.append(acc_normal)
        del error_mask_new
        Redes.append('ResNet')

    Acc_ResNet=pd.DataFrame(Accs_R)
    print('Acc_ResNet',Acc_ResNet)
    print('All_acc',All_acc)


    print(str()+' operación completada ResNet: ', datetime.now().strftime("%H:%M:%S"))


    Df_redes=pd.DataFrame(Redes)
    Df_Tecn=pd.DataFrame(Tecnicas*4)
    Df_acc=pd.DataFrame(All_acc)
    Df_acc_normal=pd.DataFrame(All_acc_normal)


    analize_by_part_Mors = pd.concat([Df_redes,Df_Tecn,Df_acc_normal,Df_acc], axis=1, join='outer')
    analize_by_part_Mors.columns =['Redes','Tecnic', 'acc_normal','acc']
    print(analize_by_part_Mors)
    analize_by_part_Mors.to_excel('MoRS/Modelo3_col_8_' + str(vol) + '/Analisis_Resultados/Palabras_x_tipo/Mask_less_LHO_L_F(HO)/ACC_Palabras_x_tipo_mask_' + str(inc) + '.xlsx', sheet_name='fichero_707', index=False)
    inc = inc + 1

print(str()+' operación completada: ', datetime.now().strftime("%H:%M:%S"))





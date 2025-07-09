#!/usr/bin/env python
# coding: utf-8

# ## SqueezeNet

# In[1]:

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

import os
import pickle as pickle
import tensorflow as tf
import numpy as np
#from Stats_original import WeightQuantization, IntroduceFaultsInWeights, GenerateFaultsList
from Nets_test_shift import GetNeuralNetworkModel
from Stats import WeightQuantization,IntroduceFaultsInWeights,GenerateFaultsList,CheckAccuracyAndLoss
from funciones import compilNet, same_elements,buffer_vectores,Base,IsoAECC,ECC, Flip,FlipPatch,FlipPatchBetter,ShiftMask
#from Nets_original import GetNeuralNetworkModel
from Training import GetDatasets
from Simulation import get_all_outputs
from Simulation import buffer_simulation, save_obj, load_obj
from datetime import datetime
import pandas as pd
import math


capa=[]

diff_nets_int_abs=[]
diff_nets_int=[]
def DifferenceOuts(outputs, outputs1,outputs2):
    diff_F_P = []
    diff_shift = []
    #write_layer = [2, 6, 10, 12, 16, 20, 22, 26, 30, 34, 36, 40, 44, 48, 50, 54, 58, 62, 64, 69, 73, 77]

    for index in range(0, len(outputs)):
        #if index == write_layer[ciclo]:
        #print('Capa', index, Net.layers[index].__class__.__name__)
        # a = outputs[index] == outputs1[index]
        # size_output = a.size
        # output_true = np.sum(a)
        #numero.append(index)
        capa.append(Net.layers[index].__class__.__name__)
        # print('capa', capa)
        # list_output_true.append(output_true)
        # list_size_output.append(size_output)
        # amount_dif = size_output - output_true
        # list_amount_dif.append(amount_dif)
        diff_nets_shif = np.sum(tf.math.abs(tf.math.subtract(outputs[index], outputs1[index])))
        diff_nets_FP = np.sum(tf.math.abs(tf.math.subtract(outputs[index], outputs2[index])))
        diff_F_P.append(diff_nets_FP)
        diff_shift.append(diff_nets_shif)
        sum_FP = np.sum(diff_F_P)
        sum_shift = np.sum(diff_shift)
    # si se hace pra una imagen se retorna esto
        #print('diff_nets', diff_nets)
    #     diff_F_P.append(diff_nets_FP)
    #     diff_shift.append(diff_nets_shif)
    # sum_FP = np.sum(diff_F_P)
    # sum_shift = np.sum(diff_shift)
    # diff_F_P.append(sum_FP)
    # diff_shift.append(sum_shift)
    # # print('sum_FP',sum_FP)
    # # print('sum_shift', sum_shift)
    # df_capa = pd.DataFrame(capa)
    # df_diff_F_P = pd.DataFrame(diff_F_P)
    # df_diff_shift = pd.DataFrame(diff_shift)
    # print('df_diff_F_P', df_diff_F_P)
    # print('df_diff_shift', df_diff_shift)
    # test_Mors = pd.concat([df_capa,df_diff_F_P,df_diff_shift], axis=1, join='outer')
    # test_Mors.columns = ['Capa','df_diff_F_P','I-df_diff_shift']
    # test_Mors.to_excel('MoRS/Modelo3_mas_fallos_col_8_experimentos/estatics/Squez_Test_shift.xlsx', sheet_name='fichero_707', index=False)
    # si se hace pra una imagen se retorna esto
    return sum_FP,sum_shift


values_diff_zero = []
mask = [16384]
mask=np.array(16384)
print(mask.dtype)

def DifferenceZero(outputs,locs,frac_size):



    for i, j in enumerate(outputs):
        #print(j.type)
        # print(j.size)
        # print(j.shape)
        j= j.flatten(order='F')
        # menor_j = locs < len(j)
        #affected_values= locs[menor_j]
        locs=locs[(locs < len(j))]
        #locs = locs[0:len(j)]
        #print('locs', locs)
        # print(j.size)
        # print(j.shape)
        affectedValues = np.take(j, locs)
        #print('affectedValues', affectedValues)
        #print(j)
        #print(outputs[index])
        #capa.append(Net.layers[i].__class__.__name__)
        Ogdtype = affectedValues.dtype
        #print(Ogdtype)
        shift = 2 ** (15)
        factor = 2 ** frac_size
        output = affectedValues * factor
        #print(output)
        output = output.astype(np.int32)
        #print(output.dtype)
        #print(output)
        output = np.where(np.less(output, 0), -output + shift, output)
        original = output
        #print('original', original)
        # test_noZeros = FuncionNotZero(original)
        original = np.bitwise_and(original, mask)
        # original = tf.bitwise.bitwise_and(original, 0)
        #print('original, bitwise',original)
        valores_afectados = np.not_equal(original, 0)
        #print('valores_afectados',valores_afectados)
        diff_zero_values = np.count_nonzero(valores_afectados)
        print ('diff_zero_values', diff_zero_values)
        #if diff_zero_values is None:
        if diff_zero_values == None:
            print('is none',valores_afectados)
        elif diff_zero_values> 0 :
            #print(type(diff_zero_values))
            #print('diff_zero_values',diff_zero_values)
            #print('diff_zero_values', diff_zero_values)
            values_diff_zero.append(diff_zero_values)
            #print('values_diff_zero',values_diff_zero)
            # print('tensor luego de valorar si es negativo', tensor)
            ## En Signo guardo el valor del signo de los numeros
            ##tensor de  1 y -1 y 0 en dependencia del sigo, será 0 si el número es cero
            # print('signo', signo)
            #output = tf.where(tf.greater_equal(output, shift), shift - output, output)
            #output = tf.cast(output / factor, dtype=Ogdtype)
            return diff_zero_values





#####SqueezeNet#########################################################################

#
trainBatchSize = testBatchSize = 1
_, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)

# # In[3]:
# # Numero de bits para activaciones (a) y pesos (w)
word_size  = 16
afrac_size = 9
aint_size  = 6
wfrac_size = 15
wint_size  = 0
#
# # Directorio de los pesos
#
cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'SqueezeNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir,'Weights')


acc_list = []
iterator = iter(test_dataset)

# ## Creo la red sin fallos


# buffer_size= 16777216
# buffer = np.array(['x']*(buffer_size))
error_mask_x=load_obj('Data/Fault Characterization/variante_mask_vc_707/error_mask_todo_x/vc_707/error_mask_054')
#print(error_mask_x)
#print(len(error_mask_x))
locs  = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/locs_0')
print(len(locs))

activation_aging = [True] * 22
Net = GetNeuralNetworkModel('SqueezeNet', (224, 224, 3), 8, aging_active=activation_aging,faulty_addresses=locs, masked_faults=error_mask_x,
                            word_size=word_size, frac_size=afrac_size, batch_size=testBatchSize)
Net.load_weights(wgt_dir).expect_partial()
loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
WeightQuantization(model=Net, frac_bits=wfrac_size, int_bits=wint_size)
Net.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
loss, acc = Net.evaluate(test_dataset)


####VGG16#########################################################################

# word_size  = 16
# afrac_size = 12
# aint_size  = 3
# wfrac_size = 15
# wint_size  = 0
#
# cwd = os.getcwd()
# wgt_dir = os.path.join(cwd, 'Data')
# wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
# wgt_dir = os.path.join(wgt_dir, 'VGG16')
# wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
# wgt_dir = os.path.join(wgt_dir,'Weights')
#
# activation_aging = [True] * 21
#
# Net = GetNeuralNetworkModel('VGG16', (224, 224, 3), 8, aging_active=activation_aging,faulty_addresses=locs, masked_faults=error_mask_x,
#                              word_size=word_size, frac_size=afrac_size, batch_size=testBatchSize)
# Net.load_weights(wgt_dir).expect_partial()
# loss = tf.keras.losses.CategoricalCrossentropy()
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# WeightQuantization(model=Net, frac_bits=wfrac_size, int_bits=wint_size)
# Net.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
# loss_sf, acc_sf = Net.evaluate(test_dataset)
#

# ####MobileNet#########################################################################

# word_size  = 16
# afrac_size = 11
# aint_size  = 4
# wfrac_size = 14
# wint_size  = 1
#
# # Tamaño del buffer de activaciones == al tamaño de la capa con mayor numero de activaciones (290400 pesos de 16 bits cada uno)
#
# # Directorio de los pesos
# cwd = os.getcwd()
# wgt_dir = os.path.join(cwd, 'Data')
# wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
# wgt_dir = os.path.join(wgt_dir, 'MobileNet')
# wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
# wgt_dir = os.path.join(wgt_dir, 'Weights')
#
#
#
# acc_list = []
# iterator = iter(test_dataset)
#
# # ## Creo la red sin fallos
# #
# activation_aging = [True] * 29
# Net = GetNeuralNetworkModel('MobileNet', (224, 224, 3), 8, aging_active=activation_aging, faulty_addresses=locs, masked_faults=error_mask_x,
#                              word_size=word_size, frac_size=afrac_size, batch_size=testBatchSize)
# Net.load_weights(wgt_dir).expect_partial()
# loss = tf.keras.losses.CategoricalCrossentropy()
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# WeightQuantization(model=Net, frac_bits=wfrac_size, int_bits=wint_size)
# Net.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
# loss_sf, acc_sf = Net.evaluate(test_dataset)


# ####ZFNet#########################################################################
# word_size  = 16
# afrac_size = 11
# aint_size  = 4
# wfrac_size = 14
# wint_size  = 1
#
# # Tamaño del buffer de activaciones == al tamaño de la capa con mayor numero de activaciones (290400 pesos de 16 bits cada uno)
#
# # Directorio de los pesos
# cwd = os.getcwd()
# wgt_dir = os.path.join(cwd, 'Data')
# wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
# wgt_dir = os.path.join(wgt_dir, 'ZFNet')
# wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
# wgt_dir = os.path.join(wgt_dir, 'Weights')
#
#
#
# acc_list = []
# iterator = iter(test_dataset)
#
#
# #
# activation_aging = [True] * 11
# Net = GetNeuralNetworkModel('ZFNet', (224, 224, 3), 8, aging_active=activation_aging,faulty_addresses=locs, masked_faults=error_mask_x,
#                              word_size=word_size, frac_size=afrac_size, batch_size=testBatchSize)
# Net.load_weights(wgt_dir).expect_partial()
# loss = tf.keras.losses.CategoricalCrossentropy()
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# WeightQuantization(model=Net, frac_bits=wfrac_size, int_bits=wint_size)
# Net.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
# loss, acc = Net.evaluate(test_dataset)
# print('acc ZFNet',acc )


# # ####DenseNet#########################################################################
#
# word_size  = 16
# afrac_size = 12
# aint_size  = 3
# wfrac_size = 13
# wint_size  = 2
#
#
#
# # Directorio de los pesos
# cwd = os.getcwd()
# wgt_dir = os.path.join(cwd, 'Data')
# wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
# wgt_dir = os.path.join(wgt_dir, 'DenseNet')
# wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
# wgt_dir = os.path.join(wgt_dir, 'Weights')
#
#
#
# acc_list = []
# iterator = iter(test_dataset)
#
# # ## Creo la red sin fallos
# #
# activation_aging = [True] * 188
# Net = GetNeuralNetworkModel('DenseNet', (224, 224, 3), 8, aging_active=activation_aging, faulty_addresses=locs, masked_faults=error_mask_x,
#                              word_size=word_size, frac_size=afrac_size, batch_size=testBatchSize)
# Net.load_weights(wgt_dir).expect_partial()
# loss = tf.keras.losses.CategoricalCrossentropy()
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# WeightQuantization(model=Net, frac_bits=wfrac_size, int_bits=wint_size)
# Net.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
# loss_sf, acc_sf = Net.evaluate(test_dataset)


####Alexnet#########################################################################

# word_size  = 16
# afrac_size = 11
# aint_size  = 4
# wfrac_size = 11
# wint_size  = 4
#
# trainBatchSize = testBatchSize = 1
# _, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (227, 227), 8, trainBatchSize, testBatchSize)
#
#
# cwd = os.getcwd()
# wgt_dir = os.path.join(cwd, 'Data')
# wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
# wgt_dir = os.path.join(wgt_dir, 'AlexNet')
# wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
# wgt_dir = os.path.join(wgt_dir,'Weights')
#
#
#
#
# acc_list = []
# iterator = iter(test_dataset)
#
# # ## Creo la red sin fallos
# #
# activation_aging = [True] * 11
# Net = GetNeuralNetworkModel('AlexNet', (227,227,3), 8, aging_active=activation_aging,faulty_addresses=locs, masked_faults=error_mask_x,
#                              word_size=word_size, frac_size=afrac_size, batch_size=testBatchSize)
# Net.load_weights(wgt_dir).expect_partial()
# loss = tf.keras.losses.CategoricalCrossentropy()
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# WeightQuantization(model=Net, frac_bits=wfrac_size, int_bits=wint_size)
# Net.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
# loss_sf, acc_sf = Net.evaluate(test_dataset)


### Analizar valores distintos de 0 en el bit 14 de cada valor
index = 1
diff_zero_total=[]
while index <= len(test_dataset):
    locs_numpy=np.array(locs)
    image = next(iterator)[0]
    #print('index+++++++++++++++++++++++++++', index)
    # print('imagen',index, image)
    outputs = get_all_outputs(Net, image)
    #print('outputs',outputs)
    diff_zero= DifferenceZero(outputs, locs_numpy,afrac_size)
    diff_zero_total.append(diff_zero)
    index += 1
print(len(diff_zero_total))
#print(type(diff_zero_total))
print(diff_zero_total)
n_diff_zero_total=np.array(diff_zero_total)
print(type(n_diff_zero_total))
#print(n_diff_zero_total)
total_none=np.count(n_diff_zero_total==None)
print('total_none',total_none)
sin_nan=n_diff_zero_total[n_diff_zero_total != np.array(None)]
#sin_nan= n_diff_zero_total[np.isnan(n_diff_zero_total)] = 0
sum_FP = np.sum(sin_nan)
print(sum_FP)
print(sin_nan)

# #
# #
# #
# #
# activation_aging = [True]*22



#
# index = 1
#
# while index <= len(test_dataset):
#     image = next(iterator)[0]
#     print('index+++++++++++++++++++++++++++', index)
#     # print('imagen',index, image)
#     outputs = get_all_outputs(Net, image)
#     outputs1 = get_all_outputs(Net_Shift, image)
#     outputs2 = get_all_outputs(Net_Filp_Patch, image)
#     # salidas del modelo sin fallas para la primer imagen del dataset de prueba
#     #outputs2 = get_all_outputs(Net_Shift, image)
#     # salidas del modelo con fallas para la primer imagen del dataset de prueba
#     net_FP,net_shift =DifferenceOuts(outputs, outputs1,outputs2)
#     diff_sum_Shift.append(net_shift)
#     diff_sum_FP.append(net_FP)
#     index = index + 1
# total_shift = np.sum(diff_sum_Shift)
# diff_sum_Shift.append(total_shift)
# total_F_p = np.sum(diff_sum_FP)
# diff_sum_FP.append(total_F_p)
# df_diff_shift = pd.DataFrame(diff_sum_Shift)
# df_diff_F_P = pd.DataFrame(diff_sum_FP)
# test_Mors = pd.concat([df_diff_F_P,df_diff_shift], axis=1, join='outer')
# test_Mors.columns = ['df_diff_F_P','df_diff_shift']
# test_Mors.to_excel('MoRS/Modelo3_mas_fallos_col_8_experimentos/estatics/Squez_Test_shift_all_dataset.xlsx', sheet_name='fichero_707', index=False)
#

# X = [x for x, y in test_dataset]
# print('aolo 1 vez')
# outputs = get_all_outputs(Net, X[0])
# outputs1 = get_all_outputs(Net_Shift, X[0])
# outputs2 = get_all_outputs(Net_Filp_Patch, X[0])
# # salidas del modelo sin fallas para la primer imagen del dataset de prueba
#  #outputs2 = get_all_outputs(Net_Shift, image)
# # salidas del modelo con fallas para la primer imagen del dataset de prueba
# sum_Shift, sum_FP=DifferenceOuts(outputs, outputs1,outputs2)




    # salidas del modelo sin fallas para la primer imagen del dataset de prueba
    #outputs2 = get_all_outputs(Net_Shift, image)
    # salidas del modelo con fallas para la primer imagen del dataset de prueba
#     #net_FP,net_shift =DifferenceOuts(outputs, outputs1,outputs2)
#     diff_sum_Shift.append(net_shift)
#     diff_sum_FP.append(net_FP)
#     index = index + 1
# total_shift = np.sum(diff_sum_Shift)
# diff_sum_Shift.append(total_shift)
# total_F_p = np.sum(diff_sum_FP)
# diff_sum_FP.append(total_F_p)
# df_diff_shift = pd.DataFrame(diff_sum_Shift)
# df_diff_F_P = pd.DataFrame(diff_sum_FP)
# test_Mors = pd.concat([df_diff_F_P,df_diff_shift], axis=1, join='outer')
# test_Mors.columns = ['df_diff_F_P','df_diff_shift']
# test_Mors.to_excel('MoRS/Modelo3_mas_fallos_col_8_experimentos/estatics/Squez_Test_shift_all_dataset.xlsx', sheet_name='fichero_707', index=False)

#index = index + 1

#total_diff=np.sum(differnce_list)
#print('total suma differnce',total_diff)
#df_total_diff = pd.DataFrame(total_diff)
#del differnce_list

# print('direrencias todas las redes',mean_diff_layer_softmaxnp)
# df_diff_layer_softmax = pd.DataFrame(mean_diff_layer_softmaxnp)
# df_acc_list = pd.DataFrame(acc_list)
#
# buf_same_elemen = pd.concat([df_total_diff], axis=1, join='outer')
# buf_same_elemen.columns = ['Redes','I-df_total_diff']
# buf_same_elemen.to_excel('MoRS/Modelo3_mas_fallos_col_8_experimentos/estatics/Squez_deff_net_original_F_P.xlsx', sheet_name='fichero_707', index=False)

# # with pd.ExcelWriter('SqueezeNet/métricas/SqueezeNet_diff_softmax_imagenes.xlsx') as writer:
# #     buf_same_elemen.to_excel(writer, sheet_name='softmax_exp', index=False)
# #     acc_media_comprobacion.to_excel(writer, sheet_name='acc_media_comprobacion', index=False)
#

#Calcular la diferencia entre la part eentera del modelo con fallos y sin fallos
#Cargar_errores = True






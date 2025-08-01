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
#from Nets_test_shift import GetNeuralNetworkModel
from Stats import WeightQuantization,IntroduceFaultsInWeights,GenerateFaultsList,CheckAccuracyAndLoss
from funciones import compilNet, same_elements,buffer_vectores,Base,IsoAECC,ECC, Flip,FlipPatch,FlipPatchBetter,ShiftMask,WordType
from Nets_original import GetNeuralNetworkModel
from Training import GetDatasets
from Simulation import get_all_outputs
from Simulation import buffer_simulation, save_obj, load_obj
from datetime import datetime
import pandas as pd


capa=[]
total_actvs =[]
diff_nets_int_abs=[]
diff_nets_int=[]
# def DifferenceOuts(outputs, outputs1,outputs2):
#     diff_F_P = []
#     diff_shift = []
#     list_actvs_afect_Shif =[]
#     list_actvs_afect_Flip = []
#     write_layer = [2,6,8,12, 19,23, 30,34, 41,43,47, 54,58, 65,69, 76,80 , 87,89,93, 100,103,107]
#
#     for index in range(0, len(outputs)):
#         if index in write_layer:
#             actvs = outputs[index].size
#             #print('Capa', index, Net.layers[index].__class__.__name__)
#             capa.append(Net.layers[index].__class__.__name__)
#             actvs_size_nets_shif = tf.math.subtract(outputs[index], outputs1[index])
#             diff_nets_shif = np.sum(tf.math.abs(tf.math.subtract(outputs[index], outputs1[index])))
#             actvs_size_nets_shif = tf.math.subtract(outputs[index], outputs1[index])
#             actvs_afect_diff_Shif = np.count_nonzero(actvs_size_nets_shif)
#             print('actvs_afect_shift', actvs_afect_diff_Shif)
#             diff_nets_FP = np.sum(tf.math.abs(tf.math.subtract(outputs[index], outputs2[index])))
#             actvs_size_nets_FP = tf.math.subtract(outputs[index], outputs2[index])
#             actvs_afect_diff_FP = np.count_nonzero(actvs_size_nets_FP)
#             print('actvs_afect_FP', actvs_afect_diff_FP)
#             diff_F_P.append(diff_nets_FP)
#             diff_shift.append(diff_nets_shif)
#             list_actvs_afect_Shif.append(actvs_afect_diff_Shif)
#             list_actvs_afect_Flip.append(actvs_afect_diff_FP)
#     sum_FP = np.sum(diff_F_P)
#     sum_shift = np.sum(diff_shift)
#     sum_actvs_afect_Shif = np.sum(list_actvs_afect_Shif)
#     sum_actvs_afect_Flip = np.sum(list_actvs_afect_Flip)
#     # si se hace pra una imagen se retorna esto
#         #print('diff_nets', diff_nets)
#     #     diff_F_P.append(diff_nets_FP)
#     #     diff_shift.append(diff_nets_shif)
#     # sum_FP = np.sum(diff_F_P)
#     # sum_shift = np.sum(diff_shift)
#     # diff_F_P.append(sum_FP)
#     # diff_shift.append(sum_shift)
#     # # print('sum_FP',sum_FP)
#     # # print('sum_shift', sum_shift)
#     # df_capa = pd.DataFrame(capa)
#     # df_diff_F_P = pd.DataFrame(diff_F_P)
#     # df_diff_shift = pd.DataFrame(diff_shift)
#     # print('df_diff_F_P', df_diff_F_P)
#     # print('df_diff_shift', df_diff_shift)
#     # test_Mors = pd.concat([df_capa,df_diff_F_P,df_diff_shift], axis=1, join='outer')
#     # test_Mors.columns = ['Capa','df_diff_F_P','I-df_diff_shift']
#     # test_Mors.to_excel('MoRS/Modelo3_mas_fallos_col_8_experimentos/estatics/Squez_Test_shift.xlsx', sheet_name='fichero_707', index=False)
#     # si se hace pra una imagen se retorna esto
#     return sum_FP,sum_shift,sum_actvs_afect_Shif,sum_actvs_afect_Flip
#
#

#### como la función d earriba solo que analizamos más datos
def DifferenceOuts(outputs, outputs1,outputs2,locs_patch,locs):
    diff_F_P = []
    diff_shift = []
    list_actvs_afect_Shif =[]
    list_actvs_afect_Flip = []
    write_layer = [2,6,8,12, 19,23, 30,34, 41,43,47, 54,58, 65,69, 76,80 , 87,89,93, 100,103,107]

    for index in range(0, len(outputs)):
        if index in write_layer:
            actvs = outputs[index].size
            actvs_LO = locs_patch[(locs_patch < outputs[index].size)]
            amount_actvs_lo=len(actvs_LO)
            #print('Capa', index, Net.layers[index].__class__.__name__)
            #capa.append(Net.layers[index].__class__.__name__)
            # Razón entre la diff de cada técnica entre el total de actvs que affectab la diffreneci
            # para ello calculo la difrenecia entre las variables implicada y cuento las actovaciones distinas de 0
            diff_nets_shif = np.sum(tf.math.abs(tf.math.subtract(outputs[index], outputs1[index])))
            diff_nets_FP = np.sum(tf.math.abs(tf.math.subtract(outputs[index], outputs2[index])))
            # diferencia (queria escribir un valor y se escribió otro)
            actvs_size_nets_shif = tf.math.subtract(outputs[index], outputs1[index])
            actvs_afect_diff_Shif = np.count_nonzero(actvs_size_nets_shif)
            print('actvs_afect_shift', actvs_afect_diff_Shif)
            actvs_size_nets_FP = tf.math.subtract(outputs[index], outputs2[index])
            actvs_afect_diff_FP = np.count_nonzero(actvs_size_nets_FP)
            print('actvs_afect_FP', actvs_afect_diff_FP)
            diff_F_P.append(diff_nets_FP)
            diff_shift.append(diff_nets_shif)
            list_actvs_afect_Shif.append(actvs_afect_diff_Shif)
            list_actvs_afect_Flip.append(actvs_afect_diff_FP)
    sum_FP = np.sum(diff_F_P)
    sum_shift = np.sum(diff_shift)
    sum_actvs_afect_Shif = np.sum(list_actvs_afect_Shif)
    sum_actvs_afect_Flip = np.sum(list_actvs_afect_Flip)

    return sum_FP,sum_shift,sum_actvs_afect_Shif,sum_actvs_afect_Flip





#
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


# ## Creo la red sin fallos


# buffer_size= 16777216
# buffer = np.array(['x']*(buffer_size))


activation_aging = [False] * 22
Net = GetNeuralNetworkModel('SqueezeNet', (224, 224, 3), 8, aging_active=activation_aging,  word_size=word_size, frac_size=afrac_size, batch_size=testBatchSize)
Net.load_weights(wgt_dir).expect_partial()
WeightQuantization(model=Net, frac_bits=wfrac_size, int_bits=wint_size)
loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
Net.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
loss, acc = Net.evaluate(test_dataset)


activation_aging = [True]*22

error_mask = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/error_mask_0')
locs = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/locs_0')
error_mask_new, locs,word_change = FlipPatchBetter(error_mask, locs)



F_patch = GetNeuralNetworkModel('SqueezeNet', (224, 224, 3), 8, faulty_addresses=locs, masked_faults=error_mask_new,
                             aging_active=activation_aging,word_size=word_size, frac_size=afrac_size,
                             batch_size=testBatchSize)
F_patch.load_weights(wgt_dir).expect_partial()
loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
WeightQuantization(model=F_patch, frac_bits=wfrac_size, int_bits=wint_size)
F_patch.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
loss, acc = F_patch.evaluate(test_dataset)

print('acc, f_p', acc)


#Creo la red llamando al experimento de FlipPatch
error_mask = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/error_mask_0')
locs = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/locs_0')
error_mask_new, locs,locs_patch,word_change = FlipPatch(error_mask, locs)

S_Patch = GetNeuralNetworkModel('SqueezeNet', (224, 224, 3), 8, faulty_addresses=locs, masked_faults=error_mask_new,
                             aging_active=activation_aging,word_size=word_size, frac_size=afrac_size,
                             batch_size=testBatchSize)
S_Patch.load_weights(wgt_dir).expect_partial()
loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
WeightQuantization(model=S_Patch, frac_bits=wfrac_size, int_bits=wint_size)
S_Patch.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
loss, acc = S_Patch.evaluate(test_dataset)

print('S_Patch', acc)
del error_mask_new

error_mask = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/error_mask_0')
locs = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/locs_0')
error_mask_new, locs,locs_patch,word_change = FlipPatch(error_mask, locs)
locs_LO, locs_HO, locs_H_L_O = WordType(error_mask_new, locs)

activation_aging = [True]*21

from Nets_test_shift import GetNeuralNetworkModel

###SquezNet#######################################################################################################################
Net_Shift = GetNeuralNetworkModel('SqueezeNet', (224, 224, 3), 8, faulty_addresses=locs, masked_faults=error_mask_new,
                             aging_active=activation_aging, word_size=word_size, frac_size=afrac_size,
                             batch_size=testBatchSize)
Net_Shift.load_weights(wgt_dir).expect_partial()
WeightQuantization(model=Net_Shift, frac_bits=wfrac_size, int_bits=wint_size)
loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
Net_Shift.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
loss, acc = Net_Shift.evaluate(test_dataset)
print('shift', acc)
# acc_list.append(acc)

write_layer = [2,6,8,12, 19,23, 30,34, 41,43,47, 54,58, 65,69, 76,80 , 87,89,93, 100,103,107]
differnce_list=[]
diff_sum_Shift =[]
diff_sum_FP =[]
diff_sum_SP = []
actvs_afect_Shif=[]
actvs_afect_FP=[]
actvs_afect_SP =[]
total_locs_lo = []
actvs_by_capa = []
# actvs_lo_capa = []
# actvs_lo_SP_capa=[]


for i, j in enumerate(write_layer):
    print('capas', j)
    index = 0
    stop = False
    diff_F_P = []
    diff_shift = []
    diff_S_P =[]
    list_actvs_afect_Shif = []
    list_actvs_afect_Flip = []
    list_actvs_afect_S_Patch = []
    iterator = iter(test_dataset)

    #while index <= len(test_dataset):
    while index <= 750 and stop == False :
        image = next(iterator)[0]
        print('index+++++++++++++++++++++++++++', index)
        # print('imagen',index, image)
        outputs = get_all_outputs(Net, image)
        outputs1 = get_all_outputs(S_Patch, image)
        outputs2 = get_all_outputs(F_patch, image)
        outputs3 = get_all_outputs(Net_Shift, image)
        output = outputs[j]
        output1 = outputs1[j]
        output2 = outputs2[j]
        output3 = outputs3[j]
        output.flatten(order='F')
        #print('output', output)
        output1.flatten(order='F')
        #print('output', output1)
        output2.flatten(order='F')
        #print('output', output2)
        output3.flatten(order='F')
        locs_lo = np.array(locs_LO)
        print('locs_lo',locs_lo)
        # locs_patch = np.array(locs_patch)
        actvs = output.size
        print('tamaño de las actvs',actvs)
        locs_affected = locs_lo[(locs_lo < output.size)]
        locs_size = len(locs_affected)
        print('tamaño de locs_ffected', locs_size)
        # actvs_LO = locs_patch[(locs_patch < actvs)]
        # amount_actvs_lo = len(actvs_LO)

        # actvs_LO_SP = locs_patch_SP[(locs_patch_SP < actvs)]
        # amount_actvs_lo_SP = len(actvs_LO_SP)
        affectedValues = np.take(output, locs_affected)
        #affectedValues= tf.gather_nd(output, locs_affected)
        #print('affectedValues',affectedValues)
        affectedValues_SPatch = np.take(output1, locs_affected)
        affectedValues_Fpatch = np.take(output2, locs_affected)
        affectedValues_Net_Shift = np.take(output3, locs_affected)
        # Razón entre la diff de cada técnica entre el total de actvs que affectab la diffreneci
        # para ello calculo la difrenecia entre las variables implicada y cuento las actovaciones distinas de 0
        #diff_nets_shif = np.sum(tf.math.abs(tf.math.subtract(output, output1)))
##operaciones para todas las activaciones
#         diff_nets_S_patch = np.sum(np.abs(np.subtract(output, output1)))
#         diff_nets_F_P = np.sum(np.abs(np.subtract(output, output2)))
#         diff_nets_Shift = np.sum(np.abs(np.subtract(output, output3)))
##Opreaciones solo para las LO afectadas
        diff_nets_S_patch = np.sum(np.abs(np.subtract(affectedValues, affectedValues_SPatch)))
        diff_nets_F_P = np.sum(np.abs(np.subtract(affectedValues, affectedValues_Fpatch)))
        diff_nets_Shift = np.sum(np.abs(np.subtract(affectedValues, affectedValues_Net_Shift)))

        # diferencia (queria escribir un valor y se escribió otro)
        actvs_size_nets_S_patch = np.subtract(affectedValues, affectedValues_SPatch)
        actvs_afect_diff_S_patch = np.count_nonzero(actvs_size_nets_S_patch)
        actvs_size_nets_FP = np.subtract(affectedValues, affectedValues_Fpatch)
        actvs_afect_diff_FP = np.count_nonzero(actvs_size_nets_FP)
        actvs_size_nets_Shift = np.subtract(affectedValues, affectedValues_Net_Shift)
        actvs_afect_diff_Shif = np.count_nonzero(actvs_size_nets_Shift)
##operaciones para todas las activaciones
        # actvs_size_nets_S_patch = np.subtract (output, output1)
        # actvs_afect_diff_S_patch = np.count_nonzero(actvs_size_nets_S_patch)
        # actvs_size_nets_FP = np.subtract(output, output2)
        # actvs_afect_diff_FP = np.count_nonzero(actvs_size_nets_FP)
        # actvs_size_nets_Shift = np.subtract(output, output3)
        # actvs_afect_diff_Shif = np.count_nonzero(actvs_size_nets_Shift)

        diff_S_P.append(diff_nets_S_patch)
        #print('lita', diff_F_P)
        diff_F_P.append(diff_nets_F_P)
        #print('lita', diff_shift)
        diff_shift.append(diff_nets_Shift)
        list_actvs_afect_S_Patch.append(actvs_afect_diff_S_patch)
        list_actvs_afect_Flip.append(actvs_afect_diff_FP)
        #print('list_actvs_afect_Flip', list_actvs_afect_Flip)
        list_actvs_afect_Shif.append(actvs_afect_diff_Shif)
        # print('list_actvs_afect_Shif', list_actvs_afect_Shif)
        index = index + 1
        if index == 750:
            stop = True
            sum_SP = np.sum(diff_S_P)
            sum_FP = np.sum(diff_F_P)
            sum_shift = np.sum(diff_shift)
            sum_actvs_afect_S_patch = np.sum(list_actvs_afect_S_Patch)
            # print('sum_actvs_afect_Flip', sum_actvs_afect_Flip)
            sum_actvs_afect_Flip = np.sum(list_actvs_afect_Flip)
            # print('sum_actvs_afect_Flip', sum_actvs_afect_Flip)
            sum_actvs_afect_Shif = np.sum(list_actvs_afect_Shif)
            #print('sum_actvs_afect_Shif',sum_actvs_afect_Shif)
            diff_sum_SP.append(sum_SP)
            diff_sum_FP.append(sum_FP)
            diff_sum_Shift.append(sum_shift)
            actvs_afect_SP.append(sum_actvs_afect_S_patch)
            actvs_afect_FP.append(sum_actvs_afect_Flip)
            actvs_afect_Shif.append(sum_actvs_afect_Shif)
            capa.append(Net.layers[j].__class__.__name__)
            actvs_by_capa.append(actvs)
            total_locs_lo.append(locs_size)
            # actvs_lo_capa.append(amount_actvs_lo)
            # actvs_lo_SP_capa.append(amount_actvs_lo_SP)
        else:
            stop = False



## Sumo todo par acalcular el valor total de la suma de todas las diferencias  par aluego agregarlo a la lista al final
total_S_p = np.sum(diff_sum_SP)
total_F_p = np.sum(diff_sum_FP)
total_shift = np.sum(diff_sum_Shift)
## Sumo el toral de activaciones por fallos para la técnica correspondiente par aluego calcular el ratio
#sum_actvs_afect_Shif = np.sum(actvs_afect_Shif)
diff_sum_SP.append(total_S_p)
diff_sum_Shift.append(total_shift)
diff_sum_FP.append(total_F_p)
capa.append('Total')

actvs_by_capa.append(np.sum(actvs_by_capa))
total_locs_lo.append(np.sum(total_locs_lo))
#actvs_lo_SP_capa.append(np.sum(actvs_lo_capa))
actvs_afect_SP.append(np.sum(actvs_afect_SP))
actvs_afect_FP.append(np.sum(actvs_afect_FP))
actvs_afect_Shif.append(np.sum(actvs_afect_Shif))


df_diff_S_P = pd.DataFrame(diff_sum_SP)
df_diff_F_P = pd.DataFrame(diff_sum_FP)
df_diff_shift = pd.DataFrame(diff_sum_Shift)


df_actvs_SP = pd.DataFrame(actvs_afect_SP)
df_actvs_FP = pd.DataFrame(actvs_afect_FP)
df_actvs_Shit = pd.DataFrame(actvs_afect_Shif)

df_capa = pd.DataFrame(capa)
df_capa_actvs = pd.DataFrame(actvs_by_capa)
df_actvs_lo_capa = pd.DataFrame(total_locs_lo)
#df_actvs_lo_capa = pd.DataFrame(actvs_lo_capa)
test_Mors = pd.concat([df_capa,df_capa_actvs, df_actvs_lo_capa,df_diff_F_P,df_actvs_FP, df_diff_S_P,   df_actvs_SP, df_diff_shift,df_actvs_Shit], axis=1, join='outer')
test_Mors.columns = ['capa',    'Total_ctvs','df_actvs_lo','diff_F_P','actvs_FP','df_diff_S_P', 'Actvs_S_P', 'diff_shift','actvs_Shit']
print(test_Mors)
test_Mors.to_excel('MoRS/Modelo3_mas_fallos_col_8_experimentos/estatics/SqueezNet_ratio_by_layers_by_technique_test_solo LO.xlsx', sheet_name='fichero_707', index=False)





#
# write_layer = [2,6,8,12, 19,23, 30,34, 41,43,47, 54,58, 65,69, 76,80 , 87,89,93, 100,103,107]
#
# differnce_list=[]
# diff_sum_Shift =[]
# diff_sum_FP =[]
# actvs_afect_Shif=[]
# actvs_afect_FP=[]
# actvs_by_capa = []
# actvs_lo_capa = []
# for i, j in enumerate(write_layer):
#     print('capas', j)
#     index = 0
#     stop = False
#     diff_F_P = []
#     diff_shift = []
#     list_actvs_afect_Shif = []
#     list_actvs_afect_Flip = []
#     iterator = iter(test_dataset)
#
#     #while index <= len(test_dataset):
#     while index <= 750 and stop == False :
#         image = next(iterator)[0]
#         print('index+++++++++++++++++++++++++++', index)
#         # print('imagen',index, image)
#         outputs = get_all_outputs(Net, image)
#         outputs1 = get_all_outputs(Net_Shift, image)
#         outputs2 = get_all_outputs(Net_Filp_Patch, image)
#         output = outputs[j]
#         output1 = outputs1[j]
#         output2 = outputs2[j]
#         output.flatten(order='F')
#         #print('output', output)
#         output1.flatten(order='F')
#         #print('output', output1)
#         output2.flatten(order='F')
#         #print('output', output2)
#         locs_patch = np.array(locs_patch)
#         actvs = output.size
#         print('tamaño de las actvs',actvs)
#         actvs_LO = locs_patch[(locs_patch < actvs)]
#         amount_actvs_lo = len(actvs_LO)
#         print('actuv lo afectadas en la capa',amount_actvs_lo )
#         # print('Capa', index, Net.layers[index].__class__.__name__)
#         # Razón entre la diff de cada técnica entre el total de actvs que affectab la diffreneci
#         # para ello calculo la difrenecia entre las variables implicada y cuento las actovaciones distinas de 0
#         #diff_nets_shif = np.sum(tf.math.abs(tf.math.subtract(output, output1)))
#         diff_nets_shif = np.sum(np.abs(np.subtract(output, output1)))
#         print('diff_nets_shif',diff_nets_shif)
#         diff_nets_FP = np.sum(np.abs(np.subtract(output, output2)))
#         print('diff_nets_FP', diff_nets_FP)
#         # diferencia (queria escribir un valor y se escribió otro)
#         actvs_size_nets_shif = np.subtract (output, output1)
#         actvs_afect_diff_Shif = np.count_nonzero(actvs_size_nets_shif)
#         print('actvs_afect_shift', actvs_afect_diff_Shif)
#         actvs_size_nets_FP = np.subtract(output, output2)
#         actvs_afect_diff_FP = np.count_nonzero(actvs_size_nets_FP)
#         print('actvs_afect_FP', actvs_afect_diff_FP)
#         diff_F_P.append(diff_nets_FP)
#         print('lita', diff_F_P)
#         diff_shift.append(diff_nets_shif)
#         print('lita', diff_shift)
#         list_actvs_afect_Shif.append(actvs_afect_diff_Shif)
#         print('list_actvs_afect_Shif', list_actvs_afect_Shif)
#         list_actvs_afect_Flip.append(actvs_afect_diff_FP)
#         print('list_actvs_afect_Flip', list_actvs_afect_Flip)
#         index = index + 1
#         if index == 750:
#             stop = True
#             sum_FP = np.sum(diff_F_P)
#             sum_shift = np.sum(diff_shift)
#             sum_actvs_afect_Shif = np.sum(list_actvs_afect_Shif)
#             print('sum_actvs_afect_Shif',sum_actvs_afect_Shif)
#             sum_actvs_afect_Flip = np.sum(list_actvs_afect_Flip)
#             print('sum_actvs_afect_Flip', sum_actvs_afect_Flip)
#             diff_sum_Shift.append(sum_shift)
#             diff_sum_FP.append(sum_FP)
#             actvs_afect_Shif.append(sum_actvs_afect_Shif)
#             actvs_afect_FP.append(sum_actvs_afect_Flip)
#             capa.append(Net.layers[j].__class__.__name__)
#             actvs_by_capa.append(actvs)
#             actvs_lo_capa.append(amount_actvs_lo)
#         else:
#             stop = False
#
#
# ## Sumo todo par acalcular el valor total de la suma de todas las diferencias  par aluego agregarlo a la lista al final
# total_shift = np.sum(diff_sum_Shift)
# ## Sumo el toral de activaciones por fallos par ala técnica correspondiente par aluego calcular el ratio
# sum_actvs_afect_Shif = np.sum(actvs_afect_Shif)
# diff_sum_Shift.append(total_shift)
# ## Sumo todo par acalcular el valor total de la suma de todas las diferencias  par aluego agregarlo a la lista al final
# total_F_p = np.sum(diff_sum_FP)
# ## Sumo el toral de activaciones por fallos par ala técnica correspondiente par aluego calcular el ratio
# diff_sum_FP.append(total_F_p)
# capa.append('Total')
# actvs_by_capa.append(np.sum(actvs_by_capa))
# actvs_lo_capa.append(np.sum(actvs_lo_capa))
# actvs_afect_FP.append(np.sum(actvs_afect_FP))
# actvs_afect_Shif.append(np.sum(actvs_afect_Shif))
#
#
# df_diff_shift = pd.DataFrame(diff_sum_Shift)
# df_diff_F_P = pd.DataFrame(diff_sum_FP)
# df_actvs_Shit = pd.DataFrame(actvs_afect_Shif)
# df_actvs_FP = pd.DataFrame(actvs_afect_FP)
# df_capa = pd.DataFrame(capa)
# df_capa_actvs = pd.DataFrame(actvs_by_capa)
# df_actvs_lo_capa = pd.DataFrame(actvs_lo_capa)
# test_Mors = pd.concat([df_capa,df_capa_actvs,df_actvs_lo_capa,df_diff_F_P,df_actvs_FP,df_diff_shift,df_actvs_Shit], axis=1, join='outer')
# test_Mors.columns = ['capa','Total_ctvs','Actvs_lo_capa','diff_F_P','actvs_FP','diff_shift','actvs_Shit']
# print(test_Mors)
# test_Mors.to_excel('MoRS/Modelo3_mas_fallos_col_8_experimentos/estatics/ratio_by_layers_Test_shift_all_dataset.xlsx', sheet_name='fichero_707', index=False)

#####Si quiero recorrerlo al revés por imagen y luego por capas mandandolo a la función de arriba

# index = 1
#
# #while index <= len(test_dataset):
# while index <= 5:
#     image = next(iterator)[0]
#     print('index+++++++++++++++++++++++++++', index)
#     # print('imagen',index, image)
#     outputs = get_all_outputs(Net, image)
#     outputs1 = get_all_outputs(Net_Shift, image)
#     outputs2 = get_all_outputs(Net_Filp_Patch, image)
#     # salidas del modelo sin fallas para la primer imagen del dataset de prueba
#     #outputs2 = get_all_outputs(Net_Shift, image)
#     # salidas del modelo con fallas para la primer imagen del dataset de prueba
#     net_FP,net_shift,actvs_afect_diff_Shif,actvs_afect_diff_FP =DifferenceOuts(outputs, outputs1,outputs2,locs_patch,locs)
#
#     diff_sum_Shift.append(net_shift)
#     diff_sum_FP.append(net_FP)
#     actvs_afect_Shif.append(actvs_afect_diff_Shif)
#     actvs_afect_FP.append(actvs_afect_diff_FP)
#
#     index = index + 1
# ## Sumo todo par acalcular el valor total de la suma de todas las diferencias  par aluego agregarlo a la lista al final
# total_shift = np.sum(diff_sum_Shift)
# ## Sumo el toral de activaciones por fallos par ala técnica correspondiente par aluego calcular el ratio
# sum_actvs_afect_Shif = np.sum(actvs_afect_Shif)
# diff_sum_Shift.append(total_shift)
# ## Sumo todo par acalcular el valor total de la suma de todas las diferencias  par aluego agregarlo a la lista al final
# total_F_p = np.sum(diff_sum_FP)
# ## Sumo el toral de activaciones por fallos par ala técnica correspondiente par aluego calcular el ratio
# sum_actvs_afect_FP = np.sum(actvs_afect_FP)
# diff_sum_FP.append(total_F_p)
# #calculo el ratio
# ratio_shfit=total_shift/sum_actvs_afect_Shif
# ratio_FP=total_F_p/sum_actvs_afect_FP
#
# #agrego el ratio al final de todo
# actvs_afect_Shif.append(ratio_shfit)
# actvs_afect_FP.append(ratio_FP)
# df_diff_shift = pd.DataFrame(diff_sum_Shift)
# df_diff_F_P = pd.DataFrame(diff_sum_FP)
#
# df_actvs_Shit = pd.DataFrame(actvs_afect_Shif)
# df_actvs_FP = pd.DataFrame(actvs_afect_FP)
# test_Mors = pd.concat([df_diff_F_P,df_actvs_FP,df_diff_shift,df_actvs_Shit], axis=1, join='outer')
# test_Mors.columns = ['df_diff_F_P','actvs_FP','df_diff_shift','actvs_Shit']
# print(test_Mors)
# test_Mors.to_excel('MoRS/Modelo3_mas_fallos_col_8_experimentos/estatics/Squez_ratio_Test_shift_all_dataset.xlsx', sheet_name='fichero_707', index=False)
# #

# X = [x for x, y in test_dataset]
# print('aolo 1 vez')
# outputs = get_all_outputs(Net, X[0])
# outputs1 = get_all_outputs(Net_Shift, X[0])
# outputs2 = get_all_outputs(Net_Filp_Patch, X[0])
# # salidas del modelo sin fallas para la primer imagen del dataset de prueba
#  #outputs2 = get_all_outputs(Net_Shift, image)
# # salidas del modelo con fallas para la primer imagen del dataset de prueba
# sum_Shift, sum_FP=DifferenceOuts(outputs, outputs1,outputs2)











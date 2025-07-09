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


total_locs=[]
values_diff_zero = []
mask = [16384]
mask=np.array(16384)
print(mask.dtype)
#### solo par alas capas qu eescriben por cada red
## acumular los locs y luego hacer la razon entre el total diferente de cero y los locs afectados
def DifferenceZero(outputs,locs,frac_size,write_layer):
    #print('tamaño el outpust',len(outputs))
    #write_layer = [2,6,8,12, 19,23, 30,34, 41,43,47, 54,58, 65,69, 76,80 , 87,89,93, 100,103,107]


    for i, j in enumerate(outputs):
        if index == write_layer[ciclo]:
            j= j.flatten(order='F')
            locs=locs[(locs < len(j))]
            locs_size=len(locs)
            print('locs_size',locs_size)
            affectedValues = np.take(j, locs)
            capa.append(Net.layers[i].__class__.__name__)
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
            original = np.bitwise_and(original, mask)
            valores_afectados = np.not_equal(original, 0)
            diff_zero_values = np.count_nonzero(valores_afectados)
            print ('diff_zero_values', diff_zero_values)
            #if diff_zero_values is None:
            if diff_zero_values is None:
                print('capa', Net.layers[i].__class__.__name__)
                print(diff_zero_values)
            else:
                    if diff_zero_values> 0  :
                        print('diff_zero_values', diff_zero_values)

                        print('capa',Net.layers[i].__class__.__name__)
                        #print(type(diff_zero_values))
                        #print('diff_zero_values',diff_zero_values)
                        #print('diff_zero_values', diff_zero_values)
                        values_diff_zero.append(diff_zero_values)
                        #print('values_diff_zero',values_diff_zero)
                        #output = tf.where(tf.greater_equal(output, shift), shift - output, output)
                        #output = tf.cast(output / factor, dtype=Ogdtype)
                        return diff_zero_values,locs_size
        else:
            return 0,0






error_mask_x=load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/error_mask_all_x')
#print(error_mask_x)
locs  = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/locs_0')
print(len(error_mask_x))
print(len(locs))
#####SqueezeNet#########################################################################

#
trainBatchSize = testBatchSize = 1
_, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)
# #
# # # # In[3]:
# # # # Numero de bits para activaciones (a) y pesos (w)
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

diff_zero_total=[]
total_locs=[]
activation_aging = [True] * 22
write_layer = [2,6,8,12, 19,23, 30,34, 41,43,47, 54,58, 65,69, 76,80 , 87,89,93, 100,103,107]
for i, valor in enumerate(activation_aging):
    ciclo=i

    Net = GetNeuralNetworkModel('SqueezeNet', (224, 224, 3), 8, aging_active=activation_aging,faulty_addresses=locs, masked_faults=error_mask_x,
                            word_size=word_size, frac_size=afrac_size, batch_size=testBatchSize)
    Net.load_weights(wgt_dir).expect_partial()
    loss = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    WeightQuantization(model=Net, frac_bits=wfrac_size, int_bits=wint_size)
    Net.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
    loss, acc = Net.evaluate(test_dataset)





### Analizar valores distintos de 0 en el bit 14 de cada valor
    index = 1


    while index <= len(test_dataset):
        locs_numpy=np.array(locs)
        image = next(iterator)[0]
        #print('index+++++++++++++++++++++++++++', index)
        # print('imagen',index, image)
        outputs = get_all_outputs(Net, image)
        #print('outputs',outputs)
        diff_zero,len_locs= DifferenceZero(outputs, locs_numpy,afrac_size,write_layer,ciclo)
        #diff_zero = DifferenceZero(outputs, locs_numpy, afrac_size)
        #total_locs.append(len_locs)
        #print('diff_zero',diff_zero)
        if diff_zero is not None and diff_zero!= '0' :
            print('diff_zero', diff_zero)
            diff_zero_total.append(diff_zero)
            total_locs.append(len_locs)
        index += 1
print(len(diff_zero_total))
print('diff_zero_total',diff_zero_total)
print('total_locs',total_locs)
n_diff_zero_total=np.array(diff_zero_total)
sin_nan=n_diff_zero_total[n_diff_zero_total != np.array(None)]
print('cantidad vaoles',len(sin_nan))
total_non=len(n_diff_zero_total)-len(sin_nan)
print('total de None', total_non)
#sin_nan= n_diff_zero_total[np.isnan(n_diff_zero_total)] = 0
sum_FP = np.sum(sin_nan)
sum_locs_afected=np.sum(total_locs)
ratio=sum_FP/sum_locs_afected
print('sum_FP',sum_FP)
print('sum_locs_afected',sum_locs_afected)
print('ratio',ratio)










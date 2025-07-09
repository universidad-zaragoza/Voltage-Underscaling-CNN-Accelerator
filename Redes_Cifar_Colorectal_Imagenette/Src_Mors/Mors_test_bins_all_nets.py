#!/usr/bin/env python
# coding: utf-8


# In[1]:

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

import os
import pickle as pickle
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.optimizers import Adam
import numpy as np
#from Nets_test_shift import GetNeuralNetworkModel
from funciones import compilNet, same_elements,buffer_vectores,Base,IsoAECC,ECC, Flip,FlipPatchBetter,ShiftMask,WordType,TestBins,ScratchPad,TestBinsAllActvs
from Stats import WeightQuantization, IntroduceFaultsInWeights, GenerateFaultsList
from Nets import GetNeuralNetworkModel
from Training import GetDatasets
from Simulation import get_all_outputs
from Simulation import buffer_simulation, save_obj, load_obj
from datetime import datetime
import pandas as pd
from datetime import datetime



#### solo par alas capas qu eescriben por cada red
## acumular los locs y luego hacer la razon entre el total diferente de cero y los locs afectados
# def DifferenceZero(output,locs,frac_size):
#     print('ciclo')
#     #print('tamaño el outpust',len(outputs))
#     #write_layer = [2,6,8,12, 19,23, 30,34, 41,43,47, 54,58, 65,69, 76,80 , 87,89,93, 100,103,107]
#
#     output.flatten(order='F')
#     locs=locs[(locs < len(j))]
#     locs_size = len(locs)
#     total_locs.append(locs_size)
#     print('locs_size', locs_size)
#     affectedValues = np.take(j, locs)
#     capa.append(Net.layers[i].__class__.__name__)
#     #locs_by_layer.append(locs_size)
#     Ogdtype = affectedValues.dtype
#     # print(Ogdtype)
#     shift = 2 ** (15)
#     factor = 2 ** frac_size
#     output = affectedValues * factor
#     output = output.astype(np.int32)
#     output = np.where(np.less(output, 0), -output + shift, output)
#     original = output
#     original = np.bitwise_and(original, mask)
#     valores_afectados = np.not_equal(original, 0)
#     diff_zero_values = np.count_nonzero(valores_afectados)
#     print('diff_zero_values', diff_zero_values)
#     # if diff_zero_values is None:
#     #return sum_values_diff_zero, sum_total_locs
#






vol=0.51
inc= 0

test= pd.DataFrame(['INTERVALO', 'CONTADOR'])
print(test)
test.to_excel('MoRS/Modelo3_col_8_'+ str(vol)+'/Analisis_Resultados/bins/test'+ str(vol)+'_'+ str(inc) + '.xlsx', sheet_name='fichero_707', index=False)

# error_mask = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/error_mask_' + str(inc))
# locs = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/locs_' + str(inc))
error_mask =load_obj('MoRS/Modelo3_col_8_'+ str(vol)+'/mask/error_mask_' + str(inc))
locs = load_obj('MoRS/Modelo3_col_8_'+ str(vol)+'/mask/locs_' + str(inc))
locs_LO, locs_HO, locs_H_L_O = WordType(error_mask, locs)



(train_ds, validation_ds, test_ds), info = tfds.load(
    "colorectal_histology",
    split=["train[:85%]", "train[85%:95%]", "train[95%:]"],
    with_info=True,
    as_supervised=True,
    shuffle_files= True,
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


capa=[]


total_locs=[]
values_diff_zero = []
mask = [16384]
mask=np.array(16384)
print(mask.dtype)



#
# # Directorio de los pesos
#
wgt_dir= ('../weights/VGG19/weights.data')



acc_list = []
iterator = iter(test_ds)

# ## Creo la red sin fallos

diff_zero_total=[]
total_locs=[]
activation_aging = [False] * 28


word_size  = 16
afrac_size = 7
aint_size  = 8
wfrac_size = 15
wint_size  = 0

Net = GetNeuralNetworkModel('VGG19', (150,150,3), 8, aging_active=activation_aging,faulty_addresses=locs, masked_faults=error_mask,
                            word_size=word_size, frac_size=afrac_size, batch_size=batch_size)
Net.load_weights(wgt_dir).expect_partial()
loss='sparse_categorical_crossentropy'
optimizer = Adam()
WeightQuantization(model=Net, frac_bits=wfrac_size, int_bits=wint_size)
Net.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
loss, acc = Net.evaluate(test_ds)
print('original', acc)




write_layer =  [2,4,7,9,11,14,16,18,20,22,25,27,29,31,33,36,38,40,42,44,47,52,56,57]


#intervalo,np_count = TestBins(write_layer,test_ds,Net,locs_LO)
intervalo,np_count= TestBinsAllActvs(write_layer,test_ds,Net)
df_intervalo= pd.DataFrame(intervalo)
df_contador= pd.DataFrame(np_count)
#print(df_contador)
df_concat= pd.concat([df_intervalo,df_contador],axis=1, join='outer')
df_concat.columns =['INTERVALO', 'CONTADOR']
print(df_concat)
df_concat.to_excel('MoRS/Modelo3_col_8_'+ str(vol)+'/Analisis_Resultados/bins/VGG19_new_bins_All_Actvs_Sinfallos'+ str(vol)+'_'+ str(inc) + '.xlsx', sheet_name='fichero_707', index=False)


#
# ####Inception####
#
#
#
#
wgt_dir= ('../weights/ResNet50/weights.data')

word_size  = 16
afrac_size = 11
aint_size  = 4
wfrac_size = 14
wint_size  = 1

activation_aging = [False] * 22

Net = GetNeuralNetworkModel('ResNet', (150,150,3), 8, faulty_addresses=locs, masked_faults=error_mask,
                             aging_active=activation_aging, word_size=word_size, frac_size=afrac_size,
                             batch_size=batch_size)
Net.load_weights(wgt_dir).expect_partial()
loss='sparse_categorical_crossentropy'
optimizer = Adam()
WeightQuantization(model=Net, frac_bits=wfrac_size, int_bits=wint_size)
Net.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
loss, acc = Net.evaluate(test_ds)


write_layer = [3, 9, 15, 20, 28, 32, 36, 40, 43, 47, 51, 55, 58, 63, 68, 76, 80, 84, 88, 91, 95, 99, 103, 106, 110,
 114, 121, 126, 131, 139, 143, 147, 151, 154, 158, 162, 166, 169, 173, 177, 181, 184, 196, 199, 211, 219,
 224, 232, 236, 240, 244, 247, 255, 259, 262,266, 268]


#intervalo,np_count = TestBins(write_layer,test_ds,Net,locs_LO)
intervalo,np_count= TestBinsAllActvs(write_layer,test_ds,Net)
df_intervalo= pd.DataFrame(intervalo)
df_contador= pd.DataFrame(np_count)
#print(df_contador)
df_concat= pd.concat([df_intervalo,df_contador],axis=1, join='outer')
df_concat.columns =['INTERVALO', 'CONTADOR']
print(df_concat)
df_concat.to_excel('MoRS/Modelo3_col_8_'+ str(vol)+'/Analisis_Resultados/bins/ResNet_new_bins_All_Actvs_Sinfallos'+ str(vol)+'_'+ str(inc) + '.xlsx', sheet_name='fichero_707', index=False)


#
word_size  = 16
afrac_size = 7
aint_size  = 8
wfrac_size = 9
wint_size  = 11

wgt_dir= ('../weights/Xception/weights.data')

activation_aging = [False] * 47

Net = GetNeuralNetworkModel('Xception', (150,150,3), 8,faulty_addresses=locs, masked_faults=error_mask, aging_active=activation_aging,
                             word_size=word_size, frac_size=afrac_size, batch_size=batch_size)
Net.load_weights(wgt_dir).expect_partial()
WeightQuantization(model=Net, frac_bits=wfrac_size, int_bits=wint_size)
loss='sparse_categorical_crossentropy'
optimizer = Adam()
Net.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
loss, acc = Net.evaluate(test_ds)
print('original', acc)


write_layer= [ 6, 11, 12, 14, 15, 21, 29, 30, 36, 44, 45, 51, 53, 60, 66, 72, 79, 85, 90, 98, 104,
110, 117, 123, 129, 136, 142, 148, 155, 161, 167, 174, 180, 186, 193, 199, 205, 212, 218, 220,
227, 232, 233, 235, 239, 240, 242]

#intervalo,np_count = TestBins(write_layer,test_ds,Net,locs_LO)
intervalo,np_count= TestBinsAllActvs(write_layer,test_ds,Net)
df_intervalo= pd.DataFrame(intervalo)
df_contador= pd.DataFrame(np_count)
#print(df_contador)
df_concat= pd.concat([df_intervalo,df_contador],axis=1, join='outer')
df_concat.columns =['INTERVALO', 'CONTADOR']
print(df_concat)
df_concat.to_excel('MoRS/Modelo3_col_8_'+ str(vol)+'/Analisis_Resultados/bins/Xception_new_bins_All_Actvs_Sinfallos'+ str(vol)+'_'+ str(inc) + '.xlsx', sheet_name='fichero_707', index=False)





wgt_dir = ('../weights/Inception/weights.data')


word_size  = 16
afrac_size = 7
aint_size  = 8
wfrac_size = 11
wint_size  = 12


activation_aging = [False] * 170

Net = GetNeuralNetworkModel('Inception', (150,150,3), 8, aging_active=activation_aging,faulty_addresses=locs, masked_faults=error_mask,
                            word_size=word_size, frac_size=afrac_size, batch_size=batch_size)
Net.load_weights(wgt_dir).expect_partial()
WeightQuantization(model=Net, frac_bits=wfrac_size, int_bits=wint_size)
loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
Net.compile(optimizer=Adam(),   loss='sparse_categorical_crossentropy',  metrics=['accuracy'])
loss, acc = Net.evaluate(test_ds)




write_layer =  [6, 11, 12, 18, 26, 27, 34, 40, 51, 52, 63, 64, 67, 73, 74, 80, 81, 90, 91, 94, 95,
116, 117, 118, 119, 126, 135, 136, 139, 140, 161, 162, 163, 164, 171, 180, 181, 183, 184, 185,
206, 207, 208, 209, 216, 225, 226, 229, 230, 251, 252, 253, 254, 261, 267, 270, 271, 280, 281,
282, 289, 295, 298, 299, 306, 307, 317, 320, 321, 327, 328, 329, 343, 344, 345, 352, 358, 369,
370, 383, 384, 406, 407, 408, 415, 421, 432, 433, 446, 447, 469, 470, 471, 478, 484, 495, 496,
509, 510, 532, 533, 534, 541, 547, 558, 559, 572, 573, 595, 596, 597, 604, 610, 621, 622, 635,
636, 658, 659, 660, 667, 673, 684, 685, 698, 699, 721, 722, 723, 730, 736, 747, 748, 761, 762,
763, 770, 776, 786, 787, 788, 797, 820, 821, 822, 823, 834, 840, 851, 852, 858, 859, 860, 861,
884, 885, 886, 887, 891, 898, 904, 915, 916, 925, 950, 951, 954, 955, 958, 962]


#intervalo,np_count = TestBins(write_layer,test_ds,Net,locs_LO)
intervalo,np_count= TestBinsAllActvs(write_layer,test_ds,Net)

df_intervalo= pd.DataFrame(intervalo)
df_contador= pd.DataFrame(np_count)
#print(df_contador)
df_concat= pd.concat([df_intervalo,df_contador],axis=1, join='outer')
df_concat.columns =['INTERVALO', 'CONTADOR']
print(df_concat)
df_concat.to_excel('MoRS/Modelo3_col_8_'+ str(vol)+'/Analisis_Resultados/bins/Inception_new_bins_All_Actvs_Sinfallos'+ str(vol)+'_'+ str(inc) + '.xlsx', sheet_name='fichero_707', index=False)










import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from Simulation import get_all_outputs
from keras import layers
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from openpyxl import Workbook
from openpyxl import load_workbook
import pathlib
from datetime import datetime
import itertools
import os
from Nets import GetNeuralNetworkModel
from Stats import WeightQuantization, ActivationStats, CheckAccuracyAndLoss, QuantizationEffect, GetReadAndWrites
from FileAnalize import analize_file, analize_file_uno,analize_file_uno_ceros, save_file, load_file
from funciones import buffer_vectores
from Simulation import save_obj, load_obj




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


word_size  = 16
afrac_size = 11
aint_size  = 4
wfrac_size = 14
wint_size  = 1


activation_aging = [False] * 22

wgt_dir= ('../weights/VGG19/weights.data')


df = QuantizationEffect('VGG19',test_ds,wgt_dir,(150,150,3),8,batch_size)
print(df)

Accs_x= []
network_size   = 290400*16   # Tamaño del buffer (en bits)
num_of_samples = 10       # Numero de muestras de distintas configuraciones de fallos a testear por cada valor de Accs/Loss

n_bits_total = np.ceil(network_size).astype(int)    #numero de bits totales

buffer = np.array(['x']*(network_size))
print(buffer)

loss, acc = CheckAccuracyAndLoss('VGG19', test_ds, wgt_dir, output_shape=8, input_shape=(150, 150, 3),
                                                    act_frac_size = 7, act_int_size = 8, wgt_frac_size = 15, wgt_int_size = 0,
                                                    batch_size=batch_size, verbose = 0, aging_active = activation_aging, weights_faults = False,
                                                     faulty_addresses=[], masked_faults = buffer)
print(acc)
Accs_x.append(acc)

# VGG19model = GetNeuralNetworkModel('VGG19',(150,150,3),8, quantization = False, aging_active=activation_aging)
# VGG19model.summary()
# #VGG19model.load_weights('../weights/VGG1950/weights.data')
# #WeightQuantization(model = VGG19model, frac_bits = wfrac_size, int_bits = wint_size)
# VGG19model.compile(  optimizer=Adam(),   loss='sparse_categorical_crossentropy',  metrics=['accuracy'])

#  b) Load/Save Weigths
#VGG19model.load_weights('../weights/VGG1950/weights.data')


# index = 1
# iterator = iter(test_ds)
# while index <= len(test_ds):
#     image = next(iterator)[0]
#     #print('index+++++++++++++++++++++++++++', index)
#     # print('imagen',index, image)
#     outputs1 = get_all_outputs(VGG19model, image)
#     index = index + 1

    #print(outputs1)
#Guardar las capas y el tamaño de estas
# X = [x for x,y in test_ds]
#         #salidas del modelo sin fallas para la primer imagen del dataset de prueba
# outputs1= get_all_outputs(VGG19model,X[0])
#
# layer_size = []
# layer_name=[]
# for index in range(0, len(outputs1)):
#
#     #print('capas', index)
#     # print('Capa',index,Net2.layers[index].__class__.__name__)
#     # a=outputs1[index]== outputs2[index]
#     if VGG19model.layers[index].__class__.__name__=='Conv2D':
#         layer_name.append(VGG19model.layers[index].__class__.__name__)
#         b = outputs1[index].size
#         print(b)
#         layer_size.append(b)
#         print('tamaño de la capa ', VGG19model.layers[index].__class__.__name__)
#         print('es', b)
# c = np.sum(layer_size)
# print('tamaño de la capa', c)
# print('total de capas', len(outputs1))
# avg_size_layers = (c / len(outputs1))
# print('avg', avg_size_layers)
# layer_size.append(c)
# layer_size.append(avg_size_layers)
#
#
# df_layer_size=pd.DataFrame(layer_size)
# df_layer_name=pd.DataFrame(layer_name)
#
# result = pd.concat([df_layer_name,df_layer_size], axis=1, join='outer')
# result.columns =['layer_name', 'layer_size']
# result.to_excel('VGG19_conv_size.xlsx', sheet_name='VGG19', index=False)
#
#
# exit()
# print(' Excel guardado: ', datetime.now().strftime("%H:%M:%S"))


# Evaluación del accuracy y loss de la red
# score = VGG19model.evaluate(test_ds, verbose=1)

#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

# for index,layer in enumerate(VGG19model.layers):
#     print(index,layer.name)








#Quantizacíón del modelo

#save_obj(df,'Data/Quantization/VGG19/Quantization_train_ds')



# num_address  = 2359808
#
# Indices      = [5,10,11,16,21,22,29,33,37,44,48,52,59,64,69,70,77,81,85,92,96,100,107,111,115,122,
#                 127,132,133,140,144,148,155,159,163,170,174,178,185,189,193,200,204,208,215,220,225,
#                 226,233,237,241,248,252,256,263,268]
# samples      = 150
#
# Data         = GetReadAndWrites(model,Indices,num_address,samples,CNN_gating=False)
# stats        = {'Lecturas': Data['Reads'],'Escrituras': Data['Writes']}
# Baseline_Acceses   = pd.DataFrame(stats).reset_index(drop=False)

# CheckAccuracyAndLoss('model', test_set, wgt_dir, act_frac_size = 12, act_int_size = 3, wgt_frac_size = 13, wgt_int_size = 2,
#                     input_shape = (224,224,3), output_shape = 8, batch_size = test_batch_size);
#
#
#
# ActivationStats(model,test_set,12,3,24)
#
#
#





network_size   = 16777216  # Tamaño del buffer (en bits)
num_of_samples = 1       # Numero de muestras de distintas configuraciones de fallos a testear por cada valor de Accs/Loss

# n_bits_total = np.ceil(network_size).astype(int)    #numero de bits totales
#
# buffer = np.array(['x']*(network_size))
# print(buffer)
# #for index in range(0,num_of_samples):
#
# loss,acc_x   = CheckAccuracyAndLoss('VGG19', test_ds, wgt_dir, output_shape=8, input_shape = (150,150,3),
#                                             act_frac_size = 11, act_int_size = 4, wgt_frac_size = 14, wgt_int_size = 1,
#                                             batch_size=batch_size, verbose = 0, aging_active = False, weights_faults = False,
#                                             masked_faults = error_mask)


# print(acc_x)
activation_aging = [True] * 22
error_mask = load_obj('Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/error_mask_058')
locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/locs_058')

loss, acc = CheckAccuracyAndLoss('VGG19', test_ds, wgt_dir, output_shape=8, input_shape=(150, 150, 3),
                                                    act_frac_size = 7, act_int_size = 8, wgt_frac_size = 15, wgt_int_size = 0,
                                                    batch_size=batch_size, verbose = 0, aging_active = activation_aging, weights_faults = False,
                                                     faulty_addresses=locs, masked_faults = error_mask)
print('parte alta', acc)


# voltaj=('0.54','0.55','0.56','0.57','0.58','0.59','0.60')
#
# Accs = []
# Accs_w = []
# Accs_a_w = []
# Loss= []
# Loss_w= []
# Loss_a_w=[]
#
# buffer_size= 16777216
# #
# # #ficheros.sort()
# #
# vol=54
# for i in range(voltaj):
#     locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/locs_0'+ str(vol))
#     error_mask = load_obj('Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/error_mask_0'+str(vol))
#
#     vol=vol+1
#
#     print('tamaño de locs', len(locs))
#     print('tamaño de error_mask', len(error_mask))
#
#
#     loss,acc   = CheckAccuracyAndLoss('VGG19', test_ds, wgt_dir, output_shape=8, input_shape = (150,150,3),
#                                             act_frac_size = 11, act_int_size = 4, wgt_frac_size = 14, wgt_int_size = 1,
#                                             batch_size=batch_size, verbose = 0, aging_active = True, weights_faults = False,
#                                             faulty_addresses = locs, masked_faults = error_mask)
#
#     loss_w,acc_w   = CheckAccuracyAndLoss('VGG19', test_ds, wgt_dir, output_shape=8, input_shape = (150,150,3),
#                                             act_frac_size = 11, act_int_size = 4, wgt_frac_size = 14, wgt_int_size = 1,
#                                             batch_size=batch_size, verbose = 0, aging_active = True, weights_faults = True,
#                                             faulty_addresses = locs, masked_faults = error_mask)
#
#     loss_a_w,acc_a_w   = CheckAccuracyAndLoss('VGG19', test_ds, wgt_dir, output_shape=8, input_shape = (150,150,3),
#                                             act_frac_size = 11, act_int_size = 4, wgt_frac_size = 14, wgt_int_size = 1,
#                                             batch_size=batch_size, verbose = 0, aging_active = False, weights_faults = True,
#                                             faulty_addresses = locs, masked_faults = error_mask)
#
#     Accs.append(acc)
#     Accs_w.append(acc_w)
#     Accs_a_w.append(acc_a_w)
#     Loss.append(loss)
#     Loss_w.append(loss_w)
#     Loss_a_w.append(loss_a_w)
#
#
#
# Acc=pd.DataFrame(Accs)
# Acc_w =pd.DataFrame(Accs_w)
# Acc_a_w =pd.DataFrame(Accs_a_w)
# Loss=pd.DataFrame(Loss)
# Loss_w.pd.DataFrame(Loss_w)
# Loss_a_w.pd.DataFrame(Loss_a_w)
#
# #Arquit=pd.DataFrame(ficheros)
# Voltajes=pd.DataFrame(voltaj)
# buf_cero = pd.concat([Voltajes,Acc,Loss,Acc_w,Loss_w, Acc_a_w,Loss_a_w], axis=1, join='outer')
# buf_cero.columns =['Voltajes','Acc','Loss', 'A_w','Loss_w' 'Acc_a_w', 'Loss_a_w']
# buf_cero.to_excel('VGG19_fichero_alterado.xlsx', sheet_name='fichero_707_054', index=False)
# print(buf_cero)
# print(' completada: ', datetime.now().strftime("%H:%M:%S"))
#

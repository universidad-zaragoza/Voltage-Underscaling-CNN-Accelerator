

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from keras import layers
from Simulation import get_all_outputs
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
afrac_size = 7
aint_size  = 8
wfrac_size = 9
wint_size  = 11


#activation_aging = [False] * 47
activation_aging = [True] * 54
#
#
# model = GetNeuralNetworkModel('Xception',(150,150,3),8, quantization = False, aging_active=activation_aging)
# model.summary()
#
# model.compile(
#     optimizer=Adam(),
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy'],
# )
#
#
#
# score = model.evaluate(train_ds, verbose=1)
#
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

wgt_dir= ('../weights/Xception/weights.data')




loss,acc_x   = CheckAccuracyAndLoss('Xception', test_ds, wgt_dir, output_shape=8, input_shape = (150,150,3),
                                            act_frac_size = 7, act_int_size = 8, wgt_frac_size = 9, wgt_int_size = 11,
                                            batch_size=batch_size, verbose = 0, aging_active = activation_aging, weights_faults = False)
#






# for index,layer in enumerate(model.layers):
#     print(index,layer.name)

# index = 1
# iterator = iter(test_ds)
# while index <= len(test_ds):
#     image = next(iterator)[0]
#     print('index+++++++++++++++++++++++++++', index)
#     # print('imagen',index, image)
#     #outputs1 = get_all_outputs(model, image)
#     index = index + 1
#     print(index)

#
# X = [x for x,y in test_ds]
#         #salidas del modelo sin fallas para la primer imagen del dataset de prueba
# outputs1= get_all_outputs(model,X[0])
# print(outputs1)

# layer_size = []
# layer_name=[]
# for index in range(0, len(outputs1)):
#     # print('Capa',index,Net2.layers[index].__class__.__name__)
#     # a=outputs1[index]== outputs2[index]
#     layer_name.append(model.layers[index].__class__.__name__)
#     b = outputs1[index].size
#     print(b)
#     layer_size.append(b)
#     print('tamaño de la capa ', model.layers[index].__class__.__name__)
#     print('es', b)
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
# result = pd.concat([df_layer_name, df_layer_size], axis=1, join='outer')
# result.columns =['layer_name', 'layer_size']
# result.to_excel('Xception_size.xlsx', sheet_name='Xception', index=False)
#

#print(' Excel guardado: ', datetime.now().strftime("%H:%M:%S"))

#exit()




#Quantizacíón del modelo
#df = QuantizationEffect('Xception',test_ds,wgt_dir,(150,150,3),8,batch_size)
#save_obj(df,'Data/Quantization/xception/Quantization_test_ds')


# num_address  = 1048576
#
# Indices      = [1,8,18,25,30,35,42,47,52,59,64,69,76,83,91,98,105,113,120,127,135,
#                 142,149,157,164,171,179,186,193,201,208,215,223,230,237,245,252,257,
#                 262,269,278]
# samples      = 150
#
# Data         = GetReadAndWrites(model,Indices,num_address,samples,CNN_gating=False)
# stats        = {'Lecturas': Data['Reads'],'Escrituras': Data['Writes']}
# Baseline_Acceses   = pd.DataFrame(stats).reset_index(drop=False)


#ActivationStats(model,test_ds,8,7,150)


Accs_x= []



network_size   = 16777216  # Tamaño del buffer (en bits)
num_of_samples = 1       # Numero de muestras de distintas configuraciones de fallos a testear por cada valor de Accs/Loss

n_bits_total = np.ceil(network_size).astype(int)    #numero de bits totales

buffer = np.array(['x']*(network_size))
print(buffer)
#for index in range(0,num_of_samples):
activation_aging = [True] * 47

loss,acc_x   = CheckAccuracyAndLoss('Xception', test_ds, wgt_dir, output_shape=8, input_shape = (150,150,3),
                                            act_frac_size = 7, act_int_size = 8, wgt_frac_size = 9, wgt_int_size = 11,
                                            batch_size=batch_size, verbose = 0, aging_active = activation_aging, weights_faults = False,
                                            masked_faults = buffer)
#
#
print('máscara de x',acc_x)
# print('loss', loss)
#
#
# loss,acc_x   = CheckAccuracyAndLoss('Xception', test_ds, wgt_dir, output_shape=8, input_shape = (150,150,3),
#                                             act_frac_size = 7, act_int_size = 8, wgt_frac_size = 7, wgt_int_size = 11,
#                                             batch_size=batch_size, verbose = 0, aging_active = False, weights_faults = False,
#                                             masked_faults = buffer)
#
#
# print(acc_x)
# print('loss', loss)
#
#
# Accs_x.append(acc_x)
#
# print('Accuracy buffer solo sin error', Accs_x)
#save_obj(Accs_x,'Data/Errors/Xception/Colorectal Dataset/Accs_x')
#save_obj(Loss,'Data/Errors/Xception/Colorectal Dataset/Loss')

locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/locs_054')
error_mask = load_obj('Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/error_mask_054')
#mask_volteada_x/mask_voltead
Accs = []
Accs_w = []
Accs_a_w = []






loss,acc   = CheckAccuracyAndLoss('Xception', test_ds, wgt_dir, output_shape=8, input_shape = (150,150,3),
                                            act_frac_size = 7, act_int_size = 8, wgt_frac_size =9, wgt_int_size = 11,
                                            batch_size=batch_size, verbose = 0, aging_active = activation_aging, weights_faults = False,
                                            faulty_addresses = locs, masked_faults = error_mask)
print('inyectando errores',acc)
print('loss', loss)
# loss,acc_w   = CheckAccuracyAndLoss('Xception', test_ds, wgt_dir, output_shape=8, input_shape = (150,150,3),
#                                             act_frac_size = 11, act_int_size = 4, wgt_frac_size = 11, wgt_int_size = 4,
#                                             batch_size=batch_size, verbose = 0, aging_active = True, weights_faults = True,
#                                             faulty_addresses = locs, masked_faults = error_mask)
#
# loss,acc_a_w   = CheckAccuracyAndLoss('Xception', test_ds, wgt_dir, output_shape=8, input_shape = (150,150,3),
#                                             act_frac_size = 11, act_int_size = 4, wgt_frac_size = 11, wgt_int_size = 4,
#                                             batch_size=batch_size, verbose = 0, aging_active = False, weights_faults = True,
#                                             faulty_addresses = locs, masked_faults = error_mask)
#
# Accs.append(acc)
# Accs_w.append(acc_w)
# Accs_a_w.append(acc_a_w)
#
#
# Acc=pd.DataFrame(Accs)
# Acc_w =pd.DataFrame(Accs_w)
# Acc_a_w =pd.DataFrame(Accs_a_w)
#
# #Arquit=pd.DataFrame(ficheros)
#
# buf_cero = pd.concat([Acc,Acc_w, Acc_a_w], axis=1, join='outer')
# buf_cero.columns =['Acc', 'A_w', 'Acc_a_w']
# buf_cero.to_excel('acc_Xception_fichero_alterado.xlsx', sheet_name='fichero_707_054', index=False)

print(' completada: ', datetime.now().strftime("%H:%M:%S"))
#!/usr/bin/env python
# coding: utf-8

# # Accesos al buffer: Lecturas y Escrituras

# In[1]:



import os
import pickle as pickle
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from Nets  import GetNeuralNetworkModel
#from Nets  import GetNeuralNetworkModel
from Stats_lect_index import WeightQuantization, ActivationStats, CheckAccuracyAndLoss, QuantizationEffect, GetReadAndWrites
from Training import GetDatasets, GetPilotNetDataset
import pandas as pd
from keras.optimizers import Adam
from datetime import datetime
from Simulation import buffer_simulation, save_obj, load_obj


vol=0.51
inc = 0

# Con el siguiente bloque obtiene  el número de lecturas y escrituras por posición de memoria tanto usando los experimentos sin usarlos

# In[2]:
error_mask = load_obj('MoRS/Modelo3_col_8_' + str(vol) + '/mask/error_mask_' + str(inc))
locs = load_obj('MoRS/Modelo3_col_8_' + str(vol) + '/mask/locs_' + str(inc))
print('error_mask', len(error_mask))
print('locs', len(locs))


print(len(locs))
print(len(error_mask))




# In[3]:


word_size  = 16
afrac_size = 11
aint_size  = 4
wfrac_size = 14
wint_size  = 1

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


wgt_dir= ('../weights/ResNet50/weights.data')

# # In[5]:
#
#


activation_aging = [True] * 22
#
# Acá la creamos, notese que como no se introduciran fallos en activaciones no es necesario pasar locs ni masks
Resnet = GetNeuralNetworkModel('ResNet', (150,150,3), 8, aging_active=activation_aging,faulty_addresses=locs, masked_faults=error_mask,
                            word_size=word_size, frac_size=afrac_size, batch_size=test_ds)
Resnet.load_weights(wgt_dir).expect_partial()
loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
WeightQuantization(model=Resnet, frac_bits=wfrac_size, int_bits=wint_size)
Resnet.compile(optimizer=Adam(),   loss='sparse_categorical_crossentropy',  metrics=['accuracy'])
loss, acc = Resnet.evaluate(test_ds)
#
#
#
#
# # # # In[7]:
# #
# #

num_address  =1048575

#num_address  = 2359808






# Indices = [0,5,10,11,16,21,22,29,33,37,44,48,52,59,64,69,70,77,81,85,92,96,100,107,111,115,122,
#                 127,132,133,140,144,148,155,159,163,170,174,178,185,189,193,200,204,208,215,220,225,
#                 226,233,237,241,248,252,256,263,268]
#
#
#
# #
#
# # #Capas con la información de procesamiento
# #
# samples      = 1 #Numero de imagenes
# # Sin Power Gating:
# Data         = GetReadAndWrites(Resnet,Indices,num_address,samples,CNN_gating=False,network_name='ResNet')
# stats        = {'Lecturas': Data['Reads'],'Escrituras': Data['Writes']}
# Baseline_Acceses   = pd.DataFrame(stats).reset_index(drop=False)
# df_writes_Read =  pd.concat([Baseline_Acceses], axis=1, join='outer')
# df_writes_Read .columns = ['index','Lecturas','Escrituras']
# print(df_writes_Read)
# print(sum(df_writes_Read['Lecturas']))
# print(sum(df_writes_Read['Escrituras']))
#save_obj(Baseline_Acceses,'ResNet/métricas/')
# with pd.ExcelWriter('ResNet/métricas/ResNet_reads_and_write_num_adress_Mors_new.xlsx') as writer:
#           df_writes_Read.to_excel(writer, sheet_name='base', index=False)
#
# # # #
# # # #
# # # # #
# # # # #
# # # # # #del Dataframe anterior obtengo uno nuevo para los índices especificados
# # # # # # VBW: Indice de las palabras que son consideradas very bad word porque aunque s ele aplique las tecnicas siguen los errores porque su estructura es xx11xx00....xx11
#VBW = [101102, 124520, 302065, 331634, 340492, 340493, 360087, 360088, 377722, 377723, 419841, 458633, 458634, 465007, 465008, 465034, 465197, 465389, 544769, 544770, 545758, 590341, 590402, 590403, 590404, 590405, 590406, 590407, 590408, 590409, 590424, 590610, 590706, 590707, 590790, 590804, 611329]
# # # # #VBW = [101102, 124520, 302065, 331634, 340492, 340493, 360087, 360088, 377722, 377723, 419841, 458633, 458634, 465007, 465008, 465034, 465197, 465389, 544769, 544770, 545758, 590341, 590402, 590403, 590404, 590405, 590406, 590407, 590408, 590409, 590424, 590610, 590706, 590707, 590790, 590804, 611329, 807694, 596228, 371814, 153245, 431863, 870432, 431606, 854134, 464132, 75599, 984684, 957428, 144540, 155294, 449309, 343438, 224550, 515399, 278216, 103082, 153193, 910945, 685437, 504077, 450176, 49594, 699634, 734341, 635221, 639518, 276509, 385631, 40833, 887620, 259787, 105166, 487082]
# # # # #VBW =[101102, 124520, 302065, 331634, 340492, 340493, 360087, 360088, 377722, 377723, 419841, 458633, 458634, 465007, 465008, 465034, 465197, 465389, 544769, 544770, 545758, 590341, 590402, 590403, 590404, 590405, 590406, 590407, 590408, 590409, 590424, 590610, 590706, 590707, 590790, 590804, 611329, 807694, 596228, 371814, 153245, 431863, 870432, 431606, 854134, 464132, 75599, 984684, 957428, 144540, 155294, 449309, 343438, 224550, 515399, 278216, 103082, 153193, 910945, 685437, 504077, 450176, 49594, 699634, 734341, 635221, 639518, 276509, 385631, 40833, 887620, 259787, 105166, 487082, 421367, 829519, 1002831, 827753, 750402, 342656, 641683, 132117, 846177, 670210, 507666, 183528, 108520, 335181, 645235, 306439, 986119, 910692, 870824, 260319, 435744, 768399, 94637, 525999, 1026694, 26510, 816438, 225078, 190841, 422703, 463124, 467197, 1030806, 379426, 871962, 746460, 360883]
# # # # VBW = [101102, 124520, 302065, 331634, 340492, 340493, 360087, 360088, 377722, 377723, 419841, 458633, 458634, 465007, 465008, 465034, 465197, 465389, 544769, 544770, 545758, 590341, 590402, 590403, 590404, 590405, 590406, 590407, 590408, 590409, 590424, 590610, 590706, 590707, 590790, 590804, 611329, 807694, 596228, 371814, 153245, 431863, 870432, 431606, 854134, 464132, 75599, 984684, 957428, 144540, 155294, 449309, 343438, 224550, 515399, 278216, 103082, 153193, 910945, 685437, 504077, 450176, 49594, 699634, 734341, 635221, 639518, 276509, 385631, 40833, 887620, 259787, 105166, 487082, 421367, 829519, 1002831, 827753, 750402, 342656, 641683, 132117, 846177, 670210, 507666, 183528, 108520, 335181, 645235, 306439, 986119, 910692, 870824, 260319, 435744, 768399, 94637, 525999, 1026694, 26510, 816438, 225078, 190841, 422703, 463124, 467197, 1030806, 379426, 871962, 746460, 360883, 971049, 559437, 989409, 145877, 845559, 1018805, 283649, 79627, 912268, 1042255, 676817, 309244, 682316, 493406, 151515, 58733, 403778, 402881, 793085, 416518, 4606, 305748, 143466, 16917, 28154, 504505, 91708, 1013618, 350501, 367555, 993020, 563837, 128, 77845, 697509, 448560, 25033]
# # # #
# # # # print('VBW',len( VBW))
# VBW = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/estatics/locs_HLO_0')
# writes_Read_VBW=df_writes_Read.iloc[VBW]
#
#
# writes_Read_VBW=df_writes_Read.iloc[VBW]
# # print('writes_Read_VBW', writes_Read_VBW)
# with pd.ExcelWriter('ResNet/métricas/ResNet_writes_Read_VBW_Mors.xlsx') as writer:
#     writes_Read_VBW.to_excel(writer, sheet_name='base', index=False)
#
# #
# #
# reads_list=np.asarray(Data['Reads'])
# k=0
# m=16
# list_values_max =[]
# for i in range(len(reads_list) // 16):
#     # print(i)
#     values_max = np.amax(reads_list[k:m])
#     if values_max!= 0:
#         list_values_max.append(values_max)
#
#
#     k = m
#     m = k + 16
# sum_values_max = np.sum(list_values_max)
# print('sum_values_max', sum_values_max)
# list_values_max.append(sum_values_max)
# print(len(list_values_max))
# #print(list_values_max)
# df_read_layers = pd.DataFrame(list_values_max)
# df_read_layers .columns = ['Lecturas x 16']
# print('máximas letcturas',df_read_layers)
# with pd.ExcelWriter('ResNet/métricas/max_lecturas_x_cada_16_direcciones_Mors.xlsx') as writer:
#          df_read_layers.to_excel(writer, sheet_name='base', index=False)


# #
# #
# #
# #
# #
#
samples = 1
#
# LI = [0,5,10,11,14,17,18,23,26,29,34,37,40,45,48,51,52,57,60,63,68,71,74,79,82,85,90,93,102,105,108,113,116,119
#      ,127,130,135,138,141,146,149,152,160,163,164,169,172,175,180,183,186,191]
# # 4 julio
# AI = [7,8,13,14,16,17,20,21,23,25,26,28,29,31,32,34,36,37,39,40,42,43,45,47,48,50,51,54,55,57,59,60,62,63,65,66,
#       68,70,71,73,74,76,77,79,81,82,84,85,87,88,90,92,93,95,96,99,100,102,104,105,107,108,110,111,113,115,116,118,119,
#       121,122,124,126,127,129,130,132,133,135,137,138,140,141,143,144,146,148,149,151,152,154,155,157,159,160,162,163,166,
#       167,169,171,172,174,175,177,178,180,182,183,185,186,188,189,191]


AI = [3, 9, 15, 20, 28, 32, 36, 40, 43, 47, 51, 55, 58, 63, 68, 76, 80, 84, 88, 91, 95, 99, 103, 106, 110,
 114, 121, 126, 131, 139, 143, 147, 151, 154, 158, 162, 166, 169, 173, 177, 181, 184, 196, 199, 211, 219,
 224, 232, 236, 240, 244, 247, 255, 259, 262,266, 268]



# #
# # #
# # #
# # #
# Buffer,ciclos =  buffer_simulation(Resnet, test_ds, integer_bits = aint_size, fractional_bits = afrac_size, samples = samples, start_from = 0,
#                                   bit_invertion = False, bit_shifting = False, CNN_gating = False,
#                                   buffer_size = 1048576, write_mode ='default', save_results = True, network_type = 'ResNet',
#                                   results_dir = 'Data/Stats/ResNet/mask_x/',
#                                    layer_indexes = LI_mio , activation_indixes = AI_mio )
#
print(str()+' operación ciclos completada: ', datetime.now().strftime("%H:%M:%S"))



import os
import pickle as pickle
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from Nets  import GetNeuralNetworkModel
from Stats_lect_index import WeightQuantization, ActivationStats, CheckAccuracyAndLoss, QuantizationEffect, GetReadAndWrites
from Training import GetDatasets, GetPilotNetDataset
import pandas as pd
from keras.optimizers import Adam
from datetime import datetime
from Simulation import buffer_simulation, save_obj, load_obj



error_mask = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/error_mask_0')
locs = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/locs_0')


word_size  = 16
afrac_size = 7
aint_size  = 8
wfrac_size = 9
wint_size  = 11

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


wgt_dir= ('../weights/Xception/weights.data')


#
activation_aging = [True] * 47
# #
# # Acá la creamos, notese que como no se introduciran fallos en activaciones no es necesario pasar locs ni masks
Xception = GetNeuralNetworkModel('Xception', (150,150,3), 8, aging_active=activation_aging,faulty_addresses=locs, masked_faults=error_mask,
                            word_size=word_size, frac_size=afrac_size, batch_size=batch_size)
Xception.load_weights(wgt_dir).expect_partial()
loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
WeightQuantization(model=Xception, frac_bits=wfrac_size, int_bits=wint_size)
Xception.compile(optimizer=Adam(),   loss='sparse_categorical_crossentropy',  metrics=['accuracy'])
loss, acc = Xception.evaluate(test_ds)


a = ActivationStats(Xception,test_ds,11,4,250)
print(a)

# #
#
#
#
num_address  =1048576
#num_address   =753048
#
#
#
# #Capas con la informacion de procesamiento
#
#
Indices      = [0,1,7,16,22,25,26,31,37,40,41,46,52,55,56,61,67,73,80,86,92,99,105,
                111,118,124,130,137,143,149,156,162,168,175,181,187,194,200,206,213,
                219,222,223,228,234,237,242]


samples      = 1 #Numero de imagenes
# Sin Power Gating:
Data         = GetReadAndWrites(Xception,Indices,num_address,samples,CNN_gating=False,network_name='Xception')
stats        = {'Lecturas': Data['Reads'],'Escrituras': Data['Writes']}
Baseline_Acceses   = pd.DataFrame(stats).reset_index(drop=False)
read = np.sum(Baseline_Acceses['Reads'])
print('total de lecturas',read)
write = np.sum(Baseline_Acceses['Writes'])
print('total de Escrituras', write)
df_writes_Read =  pd.concat([Baseline_Acceses], axis=1, join='outer')
df_writes_Read .columns = ['index','Lecturas','Escrituras']
print(df_writes_Read)
print(sum(df_writes_Read['Lecturas']))
print(sum(df_writes_Read['Escrituras']))
#save_obj(Baseline_Acceses,'Xception/métricas/Xception')
# with pd.ExcelWriter('Xception/métricas/Xception_reads_and_write_num_adress_Mors.xlsx') as writer:
#          df_writes_Read.to_excel(writer, sheet_name='base', index=False)
# #
#
# VBW = [101102, 124520, 302065, 331634, 340492, 340493, 360087, 360088, 377722, 377723, 419841, 458633, 458634, 465007, 465008, 465034, 465197, 465389, 544769, 544770, 545758, 590341, 590402, 590403, 590404, 590405, 590406, 590407, 590408, 590409, 590424, 590610, 590706, 590707, 590790, 590804, 611329]
#
VBW = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/estatics/locs_HLO_0')
#
writes_Read_VBW=df_writes_Read.iloc[VBW]


writes_Read_VBW=df_writes_Read.iloc[VBW]
# print('writes_Read_VBW', writes_Read_VBW)
with pd.ExcelWriter('Xception/métricas/Xception_writes_Read_VBW_Mors.xlsx') as writer:
    writes_Read_VBW.to_excel(writer, sheet_name='base', index=False)

#
#
reads_list=np.asarray(Data['Reads'])
k=0
m=16
list_values_max =[]
for i in range(len(reads_list) // 16):
    # print(i)
    values_max = np.amax(reads_list[k:m])
    if values_max!= 0:
        list_values_max.append(values_max)


    k = m
    m = k + 16
sum_values_max = np.sum(list_values_max)
print('sum_values_max', sum_values_max)
list_values_max.append(sum_values_max)
print(len(list_values_max))
#print(list_values_max)
df_read_layers = pd.DataFrame(list_values_max)
df_read_layers .columns = ['Lecturas x 16']
print('máximas letcturas',df_read_layers)
with pd.ExcelWriter('Xception/métricas/max_lecturas_x_cada_16_direcciones_xnew.xlsx') as writer:
         df_read_layers.to_excel(writer, sheet_name='base', index=False)



# samples = 1
# #
# # LI = [0,1,7,16,22,25,26,31,37,40,41,46,52,55,56,61,67,73,80,86,92,99,105,
# #                 111,118,124,130,137,143,149,156,162,168,175,181,187,194,200,206,213,
# #                 219,222,223,228,234,237,242]
# #

#AI = [5, 6, 11, 12, 14, 15, 21, 29, 30, 36, 44, 45, 51, 53, 60, 66, 72, 79, 85, 90, 98, 104,
# 110, 117, 123, 129, 136, 142, 148, 155, 161, 167, 174, 180, 186, 193, 199, 205, 212, 218, 220,
# 227, 232, 233, 235, 239, 240, 242]

# #
# #
# #
# Buffer,ciclos =  buffer_simulation(Xception, test_ds, integer_bits = 8, fractional_bits = 7, samples = samples, start_from = 0,
#                                   bit_invertion = False, bit_shifting = False, CNN_gating = False,
#                                   buffer_size = 1048576, write_mode ='default', save_results = True, network_type = 'Xception',
#                                   results_dir = 'Data/Stats/Xception/',
#                                   layer_indexes = LI , activation_indixes = AI)
#
# print(str()+' operación ciclos completada: ', datetime.now().strftime("%H:%M:%S"))

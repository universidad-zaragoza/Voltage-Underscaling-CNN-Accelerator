
import tensorflow as tf
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#from Heatmap_plot import Heatmap
from Training import GetIMBDDataset
from Nets_original  import GetNeuralNetworkModel
from Stats import WeightQuantization, ActivationStats, CheckAccuracyAndLoss, QuantizationEffect, GetReadAndWrites
from Simulation import buffer_simulation, save_obj, load_obj
from datetime import datetime
tf.random.set_seed(1234)
np.random.seed(1234)


trainBatchSize = testBatchSize = 1
train_set,valid_set,test_dataset  = GetIMBDDataset(trainBatchSize, testBatchSize)


activation_aging = [False]*4

SentimentalNet = GetNeuralNetworkModel('SentimentalNet',(500),1,quantization = False,aging_active = activation_aging)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
metrics = ['accuracy']
# Compile Model
SentimentalNet.compile(optimizer=optimizer, loss='binary_crossentropy', metrics = metrics)



cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'SentimentalNet')
wgt_dir = os.path.join(wgt_dir, 'IMBD Reviews Dataset')
wgt_dir = os.path.join(wgt_dir,'Weights')
SentimentalNet.load_weights(wgt_dir)

Accs=[]

ruta_bin = 'Data/Fault Characterization/error_mask_0/vc_707'
#ruta_bin = 'Data/Fault Characterization/KC705_B/RowData/'
#directorio = pathlib.Path(ruta_bin)

#ficheros = [fichero.name for fichero in directorio.iterdir()]
voltaj=('0.54','0.55','0.56','0.57','0.58','0.59','0.60')
Voltajes=pd.DataFrame(voltaj)
#
# #ficheros.sort()
#
vol=54
for i in range(1):
    error_mask = load_obj('Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/error_mask_0' + str(vol))
    locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/locs_0' + str(vol))

    vol=vol+1
    #print(i)
    #print('voltaje', vol)
    print('tamaño de locs', len(locs))
    print('tamaño de error_mask', len(error_mask))




    acc,loss   =CheckAccuracyAndLoss('SentimentalNet', test_dataset, wgt_dir, act_frac_size=15, act_int_size=0, wgt_frac_size=15,
                         wgt_int_size=0,   input_shape=(500), output_shape=1, batch_size=testBatchSize, aging_active=activation_aging)
    Accs.append(acc)
Acc_Sentim=pd.DataFrame(Accs)




buf_cero = pd.concat([Voltajes,Acc_Sentim,], axis=1, join='outer')
buf_cero.columns =['Voltajes','Sentimental']
buf_cero.to_excel('acc_base.xlsx', sheet_name='fichero_707', index=False)

print('hasta aquí bien')
print('Ejecución  completada Base: ', datetime.now().strftime("%H:%M:%S"))



print('hasta aquí bien')
Accs_ecc_4=[]
vol=54
for i in range(1):
    error_mask = load_obj('Data/Fault Characterization/variante_mask_vc_707/ECCx4x1/ECCx/vc_707/error_mask_0' + str(vol))
    locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/ECCx4x1/ECCx/vc_707/locs_0' + str(vol))

    vol=vol+1
    #print(i)
    #print('voltaje', vol)
    print('tamaño de locs', len(locs))
    print('tamaño de error_mask', len(error_mask))




    acc_ecc_4,loss   =CheckAccuracyAndLoss('SentimentalNet', test_dataset, wgt_dir, act_frac_size=15, act_int_size=0, wgt_frac_size=15,
                         wgt_int_size=0,   input_shape=(500), output_shape=1, batch_size=testBatchSize, aging_active=activation_aging)
    Accs_ecc_4.append(acc_ecc_4)
Acc_Sentim_ecc_4=pd.DataFrame(Accs_ecc_4)



buf_cero = pd.concat([Voltajes,Acc_Sentim_ecc_4,], axis=1, join='outer')
buf_cero.columns =['Voltajes','Sentimental']
buf_cero.to_excel('acc_sentime_ecc_4.xlsx', sheet_name='fichero_707', index=False)


print('Ejecución  completada ISO_A_ECC: ', datetime.now().strftime("%H:%M:%S"))
Accs_ecc=[]
vol=54
for i in range(1):
    error_mask = load_obj('Data/Fault Characterization/variante_mask_vc_707/ECC/ECC/vc_707/error_mask_0' + str(vol))
    locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/ECC/ECC/vc_707/locs_0' + str(vol))

    vol=vol+1
    #print(i)
    #print('voltaje', vol)
    print('tamaño de locs', len(locs))
    print('tamaño de error_mask', len(error_mask))




    acc_ecc,loss   =CheckAccuracyAndLoss('SentimentalNet', test_dataset, wgt_dir, act_frac_size=15, act_int_size=0, wgt_frac_size=15,
                         wgt_int_size=0,   input_shape=(500), output_shape=1, batch_size=testBatchSize, aging_active=activation_aging)
    Accs_ecc.append(acc_ecc)
Acc_Sentim_ecc=pd.DataFrame(Accs_ecc)



buf_cero = pd.concat([Voltajes,Acc_Sentim_ecc,], axis=1, join='outer')
buf_cero.columns =['Voltajes','Sentimental']
buf_cero.to_excel('acc_sentime_ecc.xlsx', sheet_name='fichero_707', index=False)

print('Ejecución  completada ECC: ', datetime.now().strftime("%H:%M:%S"))

Accs_flip=[]
vol=54
for i in range(1):
    error_mask = load_obj('Data/Fault Characterization/variante_mask_vc_707/mask_volteada_x/mask_volteada/vc_707/error_mask_0' + str(vol))
    locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/mask_volteada_x/mask_volteada/vc_707/locs_0' + str(vol))

    vol=vol+1
    #print(i)
    #print('voltaje', vol)
    print('tamaño de locs', len(locs))
    print('tamaño de error_mask', len(error_mask))




    acc_flip,loss   =CheckAccuracyAndLoss('SentimentalNet', test_dataset, wgt_dir, act_frac_size=15, act_int_size=0, wgt_frac_size=15,
                         wgt_int_size=0,   input_shape=(500), output_shape=1, batch_size=testBatchSize, aging_active=activation_aging)
    Accs_flip.append(acc_flip)
Acc_Sentim_flip=pd.DataFrame(Accs_flip)



buf_cero = pd.concat([Voltajes,Acc_Sentim_flip,], axis=1, join='outer')
buf_cero.columns =['Voltajes','Sentimental']
buf_cero.to_excel('acc_sentime_ecc.xlsx', sheet_name='fichero_707', index=False)

print('Ejecución  completada Flip: ', datetime.now().strftime("%H:%M:%S"))

Accs_f_p=[]
vol=54
for i in range(1):
    error_mask = load_obj('Data/Fault Characterization/variante_mask_vc_707/flip_patch/byte_x/vc_707/error_mask_0' + str(vol))
    locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/flip_patch/byte_x/vc_707/locs_0' + str(vol))

    vol=vol+1
    #print(i)
    #print('voltaje', vol)
    print('tamaño de locs', len(locs))
    print('tamaño de error_mask', len(error_mask))




    acc_f_p,loss   =CheckAccuracyAndLoss('SentimentalNet', test_dataset, wgt_dir, act_frac_size=15, act_int_size=0, wgt_frac_size=15,
                         wgt_int_size=0,   input_shape=(500), output_shape=1, batch_size=testBatchSize, aging_active=activation_aging)
    Accs_f_p.append(acc_f_p)
Acc_Sentim_f_p=pd.DataFrame(Accs_f_p)



buf_cero = pd.concat([Voltajes,Acc_Sentim_f_p,], axis=1, join='outer')
buf_cero.columns =['Voltajes','Sentimental']
buf_cero.to_excel('acc_sentime_ecc.xlsx', sheet_name='fichero_707', index=False)

print('Ejecución  completada Flip + Patch: ', datetime.now().strftime("%H:%M:%S"))
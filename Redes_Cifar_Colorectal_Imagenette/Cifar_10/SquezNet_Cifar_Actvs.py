#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from keras.datasets import cifar10
from Nets_mobile import GetNeuralNetworkModel
from Stats_Cifar_ import WeightQuantization, ActivationStats, CheckAccuracyAndLoss, QuantizationEffect, GetReadAndWrites
from funciones import buffer_vectores
from Simulation import save_obj, load_obj
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
import pandas as pd
from datetime import datetime
import time


# Buscar la forma de acomodar el proyecto por dataset
#d  /home/reynier/py_projects/Deployments/functions


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


(train_ds, validation_ds, test_ds), info = tfds.load('cifar10',
                                    split=["train", "test[:35%]", "test[35%:]"],
                                    as_supervised = True,
                                    with_info=True,
                                    shuffle_files= True)


size = (32, 32)
batch_size = 1

def normalize_resize(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.divide(image, 255)
    image = tf.image.resize(image, size)
    return image, label

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_saturation(image, 0.7, 1.3)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_crop(image, (32, 32, 3))
    return image, label

train_ds = train_ds.map(normalize_resize).cache().map(augment).batch(batch_size).prefetch(buffer_size=1)
validation_ds = validation_ds.map(normalize_resize).cache().batch(batch_size).prefetch(buffer_size=1)
test_ds = test_ds.map(normalize_resize).cache().batch(batch_size).prefetch(buffer_size=1)

activation_aging = [False]*29
SqueezeNet = GetNeuralNetworkModel('SqueezeNet',(32,32,3),10, quantization = False, aging_active=activation_aging)

SqueezeNet.summary()



SqueezeNet.compile(
    optimizer=Adam(0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])


wgt_dir= ('../Trained_Weights_cifar/SqueezeNet/weights.data')


#df = QuantizationEffect('SqueezeNet',test_ds,wgt_dir,(32,32,3),10,batch_size)
#
#
#quantizacion=pd.DataFrame(df)
#
#quantizacion.to_excel('quantizacion.xlsx', sheet_name='fichero_707', index=False)
#save_obj(df,'Data/Quantization/SqueezeNet_cifar/Quantization_MobileNet_cifar')

SqueezeNet.load_weights(wgt_dir)

score = SqueezeNet.evaluate(test_ds, verbose=1)
#
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#
#
#
# score = SqueezeNet.evaluate(train_ds, verbose=1)
# #
# print('train loss:', score[0])
# print('train accuracy:', score[1])

print(str()+' Quantización completada MobileNet_Cifar: ', datetime.now().strftime("%H:%M:%S"))

# ficheros = [fichero.name for fichero in directorio.iterdir()]
# voltaj=('0.54','0.55','0.56','0.57','0.58','0.59','0.60')
voltaj = [54, 55, 56, 57, 58, 59, 60]
# voltaj=[56,58]
Voltajes = pd.DataFrame(voltaj)
print(Voltajes)
paso = 1
#
# #ficheros.sort()
#
vol = voltaj[0]
Accs_B = []
Accs_I = []
Accs_E = []
Accs_F = []
Accs_F_P = []
run_time_B=[]
run_time_I=[]
run_time_E=[]
run_time_F=[]
run_time_F_P=[]




activation_aging = [True] * 29
for i in range(7):
    inicio = time.time()
    error_mask = load_obj(  'Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/error_mask_0' + str(vol))
    locs = load_obj(  'Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/locs_0' + str(vol))
    vol = vol + paso
    # print(i)
    # print('voltaje',vol)
    print('tamaño de locs', len(locs))

    loss, acc = CheckAccuracyAndLoss('SqueezeNet', test_ds, wgt_dir, output_shape=10, input_shape=(32, 32, 3),
                                     act_frac_size=11, act_int_size=4, wgt_frac_size=15, wgt_int_size=0,
                                     batch_size=1, verbose=0, aging_active=activation_aging, weights_faults=False,
                                     faulty_addresses=locs, masked_faults=error_mask)
    Accs_B.append(acc)
    fin = time.time()
    time_run = fin - inicio
    run_time_B.append(time_run)
DF_run_time_B = pd.DataFrame(run_time_B)
Base = pd.DataFrame(Accs_B)
print(Base)
#

vol = voltaj[0]

activation_aging = [True] * 29
for i in range(7):
    inicio = time.time()
    locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/ECCx4x1/ECCx/vc_707/locs_0'+ str(vol))
    error_mask = load_obj('Data/Fault Characterization/variante_mask_vc_707/ECCx4x1/ECCx/vc_707/error_mask_0'+ str(vol))
    vol = vol + paso
    print(i)
    # print('voltaje',vol)
    print('tamaño de locs', len(locs))

    loss, acc = CheckAccuracyAndLoss('SqueezeNet', test_ds, wgt_dir, output_shape=10, input_shape=(32, 32, 3),
                                     act_frac_size=11, act_int_size=4, wgt_frac_size=15, wgt_int_size=0,
                                     batch_size=1, verbose=0, aging_active=activation_aging, weights_faults=False,
                                     faulty_addresses=locs, masked_faults=error_mask)

    Accs_I.append(acc)
    fin = time.time()
    time_run = fin - inicio
    run_time_I.append(time_run)
DF_run_time_I = pd.DataFrame(run_time_I)
iso_A = pd.DataFrame(Accs_I)
print(iso_A)



vol = voltaj[0]
activation_aging = [True] * 29
for i in range(7):
    inicio = time.time()
    locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/ECC/ECC/vc_707/locs_0'+ str(vol))
    error_mask = load_obj('Data/Fault Characterization/variante_mask_vc_707/ECC/ECC/vc_707/error_mask_0'+ str(vol))

    vol = vol + paso
    # print(i)
    # print('voltaje',vol)
    print('tamaño de locs', len(locs))

    loss, acc = CheckAccuracyAndLoss('SqueezeNet', test_ds, wgt_dir, output_shape=10, input_shape=(32, 32, 3),
                                     act_frac_size=11, act_int_size=4, wgt_frac_size=15, wgt_int_size=0,
                                     batch_size=1, verbose=0, aging_active=activation_aging, weights_faults=False,
                                     faulty_addresses=locs, masked_faults=error_mask)
    Accs_E.append(acc)
    fin = time.time()
    time_run = fin - inicio
    run_time_E.append(time_run)
DF_run_time_E = pd.DataFrame(run_time_E)
ECC = pd.DataFrame(Accs_E)
print(ECC)



vol = voltaj[0]
activation_aging = [True] * 29
for i in range(7):
    inicio = time.time()
    locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/mask_volteada_x/mask_volteada/vc_707/locs_0'+ str(vol))
    error_mask = load_obj( 'Data/Fault Characterization/variante_mask_vc_707/mask_volteada_x/mask_volteada/vc_707/error_mask_0'+ str(vol))
    vol = vol + paso
    # print(i)
    # print('voltaje',vol)
    print('tamaño de locs', len(locs))

    loss, acc = CheckAccuracyAndLoss('SqueezeNet', test_ds, wgt_dir, output_shape=10, input_shape=(32, 32, 3),
                                     act_frac_size=11, act_int_size=4, wgt_frac_size=15, wgt_int_size=0,
                                     batch_size=1, verbose=0, aging_active=activation_aging, weights_faults=False,
                                     faulty_addresses=locs, masked_faults=error_mask)
    Accs_F.append(acc)
    fin = time.time()
    time_run = fin - inicio
    run_time_F.append(time_run)
DF_run_time_F = pd.DataFrame(run_time_F)
Flip = pd.DataFrame(Accs_F)
print(Flip)



vol = voltaj[0]
activation_aging = [True] * 29
for i in range(7):
    inicio = time.time()
    locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/flip_patch/byte_x/vc_707/locs_0' + str(vol))
    error_mask = load_obj('Data/Fault Characterization/variante_mask_vc_707/flip_patch/byte_x/vc_707/error_mask_0' + str(vol))
    vol = vol + paso
    # print(i)
    # print('voltaje',vol)
    print('tamaño de locs', len(locs))

    loss, acc = CheckAccuracyAndLoss('SqueezeNet', test_ds, wgt_dir, output_shape=10, input_shape=(32, 32, 3),
                                     act_frac_size=11, act_int_size=4, wgt_frac_size=15, wgt_int_size=0,
                                     batch_size=1, verbose=0, aging_active=activation_aging, weights_faults=False,
                                     faulty_addresses=locs, masked_faults=error_mask)
    Accs_F_P.append(acc)
    fin = time.time()
    time_run = fin - inicio
    run_time_F_P.append(time_run)
DF_run_time_F_P = pd.DataFrame(run_time_F_P)
F_P = pd.DataFrame(Accs_F_P)
print(F_P)
#
#
#
#
#
#
buf_cero = pd.concat( [Voltajes, Base,DF_run_time_B, iso_A, DF_run_time_I, ECC, DF_run_time_E, Flip,DF_run_time_F, F_P,DF_run_time_F_P], axis=1, join='outer')
buf_cero.columns = ['Voltajes', 'Base','Time_B', 'iso_A','Time_I', 'ECC', 'Time_E','Flip','Time_F' , 'F_P', 'Time_F_p']
buf_cero.to_excel('SqueezeNet_Cifar_all_v_all_exp.xlsx', sheet_name='fichero_707', index=False)

# buf_cero = pd.concat([Voltajes,Acc_SqueezeNet,Acc_DenseNet,], axis=1, join='outer')
# buf_cero.columns =['Voltajes', 'Acc_Sque', 'Acc_Dense']
# buf_cero.to_excel('acc_baseline_sq_dens_060.xlsx', sheet_name='fichero_707', index=False)

# print('buf_cero', buf_cero)
print(str() + ' operación completada: ', datetime.now().strftime("%H:%M:%S"))





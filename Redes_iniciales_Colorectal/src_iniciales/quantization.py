import os
import pickle as pickle
import tensorflow as tf
import numpy as np
#from Nets_test_shift import GetNeuralNetworkModel
from Stats_original import WeightQuantization,IntroduceFaultsInWeights,GenerateFaultsList,CheckAccuracyAndLoss,QuantizationEffect
from funciones import compilNet, same_elements,buffer_vectores,Base,IsoAECC,ECC, Flip,FlipPatchBetter,ShiftMask,WordType
from Nets import GetNeuralNetworkModel
from Training import GetDatasets
from Simulation import get_all_outputs
from Simulation import buffer_simulation, save_obj, load_obj
import pandas as pd
from datetime import datetime
import time

trainBatchSize = testBatchSize = 1
_, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)



cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'MobileNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir, 'Weights')


MobileNet   = GetNeuralNetworkModel('MobileNet',(224,224,3),8, quantization = False, aging_active=False)
loss      = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
MobileNet.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])





MobileNet_quiantization = QuantizationEffect('MobileNet',test_dataset,wgt_dir,(224,224,3),8,trainBatchSize)

quantizacion_MobileNet=pd.DataFrame(MobileNet_quiantization)
#
quantizacion_MobileNet.to_excel('Analizando_fichero_detalle/tamaño de las redes/quinatization/quantizacion_MobileNet.xlsx', sheet_name='fichero_707', index=False)



_, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)

cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'VGG16')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir, 'Weights')

VGG16   = GetNeuralNetworkModel('VGG16',(224,224,3),8, quantization = False, aging_active=False)
loss      = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
VGG16.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])


VGG16_quiantization = QuantizationEffect('VGG16',test_dataset,wgt_dir,(224, 224,3),8,trainBatchSize)

quantizacion_VGG16=pd.DataFrame(VGG16_quiantization)
#
quantizacion_VGG16.to_excel('Analizando_fichero_detalle/tamaño de las redes/quinatization/quantizacion_VGG16.xlsx', sheet_name='fichero_707', index=False)





_, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (227, 227), 8, trainBatchSize, testBatchSize)


cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'AlexNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir,'Weights')

activation_aging = [False] * 11

AlexNet = GetNeuralNetworkModel('AlexNet', (227,227,3), 8,quantization = False, aging_active=activation_aging)
loss      = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
AlexNet.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

#AlexNet.evaluate(test_dataset)

AlexNet_quiantization = QuantizationEffect('AlexNet',test_dataset,wgt_dir,(227,227,3),8,trainBatchSize)

quantizacion_AlexNet=pd.DataFrame(AlexNet_quiantization)
#
quantizacion_AlexNet.to_excel('Analizando_fichero_detalle/tamaño de las redes/quinatization/quantizacion_AlexNet.xlsx', sheet_name='fichero_707', index=False)






_, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)

cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'SqueezeNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir, 'Weights')

SqueezeNet     = GetNeuralNetworkModel('SqueezeNet',(224,224,3),8, quantization = False, aging_active=False)
loss      = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
SqueezeNet.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

SqueezeNet_quiantization = QuantizationEffect('SqueezeNet',test_dataset,wgt_dir,(224,224,3),8,trainBatchSize)

quantizacion_SqueezeNet=pd.DataFrame(SqueezeNet_quiantization)
#
quantizacion_SqueezeNet.to_excel('Analizando_fichero_detalle/tamaño de las redes/quinatization/quantizacion_SqueezeNet.xlsx', sheet_name='fichero_707', index=False)




cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'DenseNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir, 'Weights')

DenseNet   = GetNeuralNetworkModel('DenseNet',(224,224,3),8, quantization = False, aging_active=False)
loss      = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
DenseNet.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

DenseNet_quiantization = QuantizationEffect('DenseNet',test_dataset,wgt_dir,(224,224,3),8,trainBatchSize)

quantizacion_DenseNet=pd.DataFrame(DenseNet_quiantization)
#
quantizacion_DenseNet.to_excel('Analizando_fichero_detalle/tamaño de las redes/quinatization/quantizacion_DenseNet.xlsx', sheet_name='fichero_707', index=False)


cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'ZFNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir, 'Weights')

ZFNet   = GetNeuralNetworkModel('ZFNet',(224,224,3),8, quantization = False, aging_active=False)
loss      = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
ZFNet.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])


ZFNet_quiantization = QuantizationEffect('ZFNet',test_dataset,wgt_dir,(224,224,3),8,trainBatchSize)

quantizacion_ZFNet=pd.DataFrame(ZFNet_quiantization)
#
quantizacion_ZFNet.to_excel('Analizando_fichero_detalle/tamaño de las redes/quinatization/quantizacion_ZFNet.xlsx', sheet_name='fichero_707', index=False)

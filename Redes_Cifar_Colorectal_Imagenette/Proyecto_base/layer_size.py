
import pickle as pickle
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.optimizers import Adam
import numpy as np
from Stats_original_no_usar import WeightQuantization, IntroduceFaultsInWeights, GenerateFaultsList, CheckAccuracyAndLoss
from Nets import GetNeuralNetworkModel
from Training import GetPilotNetDataset
from Training import GetDatasets
from Simulation import get_all_outputs
from Simulation import buffer_simulation, save_obj, load_obj
from datetime import datetime
import collections
import pandas as pd
import os, sys
import pathlib




wgt_dir= ('../weights/Xception/weights.data')

(train_ds, validation_ds, test_ds), info = tfds.load(
    "colorectal_histology",
    split=["train[:85%]", "train[85%:95%]", "train[95%:]"],
    with_info=True,
    as_supervised=True,
    shuffle_files= True
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

activation_aging = [False] * 47

model = GetNeuralNetworkModel('Xception',(150,150,3),8, quantization = False, aging_active=activation_aging)
model.summary()
# #VGG19model.load_weights('../weights/VGG1950/weights.data')
# #WeightQuantization(model = VGG19model, frac_bits = wfrac_size, int_bits = wint_size)
# VGG19model.compile(  optimizer=Adam(),   loss='sparse_categorical_crossentropy',  metrics=['accuracy'])

#  b) Load/Save Weigths
#model.load_weights('../weights/VGG1950/weights.data')


# index = 1
# iterator = iter(test_ds)
# while index <= len(test_ds):
#     image = next(iterator)[0]
#     #print('index+++++++++++++++++++++++++++', index)
#     # print('imagen',index, image)
#     outputs1 = get_all_outputs(model, image)
#     index = index + 1
#
#     print(outputs1)
# 1#Guardar las capas y el tamaño de estas
# X = [x for x,y in test_ds]
#         #salidas del modelo sin fallas para la primer imagen del dataset de prueba
# outputs1= get_all_outputs(model,X[0])
#
# layer_size = []
# all_layer_size = []
# layer_name=[]
# all_layer=[]
#
# for index in range(0, len(outputs1)):
#
#     print('capas', model.layers[index].__class__.__name__)
#     all_layer.append(model.layers[index].__class__.__name__)
#     size_all_layer = outputs1[index].size
#     print(size_all_layer)
#     all_layer_size.append(size_all_layer)
#     print('tamaño de la capa ', model.layers[index].__class__.__name__)

    # # print('Capa',index,Net2.layers[index].__class__.__name__)
    # # a=outputs1[index]== outputs2[index]
    # if model.layers[index].__class__.__name__=='Conv2D':
    #     layer_name.append(model.layers[index].__class__.__name__)
    #     b = outputs1[index].size
    #
    #     layer_size.append(b)
    #     #print('tamaño de la capa ', model.layers[index].__class__.__name__)

# c = np.sum(layer_size)
# print('tamaño de la capa', c)
# print('total de capas', len(outputs1))
# avg_size_layers = (c / len(outputs1))
# print('avg', avg_size_layers)
# layer_size.append(c)
# layer_size.append(avg_size_layers)

# d = np.sum(size_all_layer)
# print('tamaño de la capa', d)
# print('total de capas', len(outputs1))
# #avg_size_layers = (d / len(outputs1))
# #print('avg', avg_size_layers)
# all_layer_size.append(d)
# #all_layer_size.append(avg_size_layers)



# df_layer_size=pd.DataFrame(layer_size)
# df_layer_name=pd.DataFrame(layer_name)

# df_all_layer_size=pd.DataFrame(all_layer_size)
# df_all_layer=pd.DataFrame(all_layer)

# result = pd.concat([df_layer_name,df_layer_size], axis=1, join='outer')
# result.columns =['layer_name', 'layer_size']
# result.to_excel('VGG_size.xlsx', sheet_name='VGG', index=False)
# all_net_size = pd.concat([df_all_layer,df_all_layer_size], axis=1, join='outer')
# all_net_size.columns =['layer_name', 'layer_size']
# all_net_size.to_excel('Xception_all_layer_size.xlsx', sheet_name='Xception', index=False)
#

#### Para Xception

input_shape = []
layer_name = []
for index, layer in enumerate(model.layers):
    layer_name.append(layer.name)
    input_shape.append(layer.input_shape[1:])
    # print(index,layer.name)
    # print(index,layer.input_shape[1:])

df_layer_name = pd.DataFrame(layer_name)
df_layer_size = pd.DataFrame(input_shape)
print(df_layer_size)
df_layer_size.to_excel('Shape_Xception.xlsx', sheet_name='resnet', index=False)
df_layer_name.to_excel('layer_name_Xception.xlsx', sheet_name='resnet', index=False)

#capas con fallos
#lamdas despues d euna convolución


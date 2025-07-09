#!/usr/bin/env python
# coding: utf-8

# ## MobileNet

# In[1]:

#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

import os
import pickle as pickle
import tensorflow as tf
import numpy as np
from Stats import WeightQuantization, IntroduceFaultsInWeights, GenerateFaultsList
from Nets_original import GetNeuralNetworkModel
from Training import GetDatasets
from Simulation import get_all_outputs
from Simulation import buffer_simulation, save_obj, load_obj
from datetime import datetime
import pandas as pd
from funciones import compilNet, same_elements


# In[2]:


trainBatchSize = testBatchSize = 1
_, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)


# In[3]:


# Numero de bits para activaciones (a) y pesos (w)
word_size  = 16
afrac_size = 11
aint_size  = 4
wfrac_size = 14
wint_size  = 1

# Tamaño del buffer de activaciones == al tamaño de la capa con mayor numero de activaciones (290400 pesos de 16 bits cada uno)
abuffer_size = 16777216
# Directorio de los pesos
cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'MobileNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir,'Weights')


# In[4]:


activation_aging = [False]*29



Net1 = GetNeuralNetworkModel('MobileNet', (224,224,3), 8, aging_active=activation_aging,
                             word_size=word_size, frac_size=afrac_size, batch_size = testBatchSize)
Net1.load_weights(wgt_dir).expect_partial()
loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
WeightQuantization(model = Net1, frac_bits = wfrac_size, int_bits = wint_size)
Net1.compile(optimizer=optimizer, loss=loss, metrics='accuracy')

loss_sf,acc_sf =Net1.evaluate(test_dataset)


list_size_output=[]
list_size_output=[]
list_output_true=[]
list_amount_dif=[]
list_ratio=[]
list_ratio_zero=[]
list_total_zero=[]
numero=[]
capa=[]
def same_elements(outputs1, outputs2, ciclo, acc_list):
    write_layer = [2,9,15,21,28,34,40,46,53,59,65,71,78,84,90,96,102,108,114,120,126,132,138,144,151,157,163,169,173,179]

    for index in range(0, len(outputs2)):
        if index == write_layer[ciclo]:
            # for i, layer in enumerate(write_layer):
            print('Capa', index, Net2.layers[index].__class__.__name__)
            a = outputs1[index] == outputs2[index]
            size_output = a.size
            output_true = np.sum(a)
            numero.append(index)
            print('numero', numero)
            capa.append(Net2.layers[index].__class__.__name__)
            print('capa', capa)
            list_output_true.append(output_true)
            list_size_output.append(size_output)
            amount_dif = size_output - output_true
            list_amount_dif.append(amount_dif)
            ratio = ((amount_dif * 100) / size_output)
            list_ratio.append(ratio)
            non_zero = (np.count_nonzero(outputs2[index]))
            total_zero = size_output - non_zero
            ratio_zero = ((total_zero * 100) / size_output)
            list_ratio_zero.append(ratio_zero)
            list_total_zero.append(total_zero)
    return (numero, capa, list_size_output, list_amount_dif, list_ratio, list_total_zero, list_ratio_zero)


def SofmaxDiffElements(outputs1, outputs2, acc_list):
    list_size_output = []
    list_size_output = []
    list_output_true = []
    list_amount_dif = []
    numero = []
    capa = []
    diff_nets_int = []
    diff_nets_int_abs = []

    outp_1 = []
    outp_2 = []


    for index in range(0, len(outputs2)):

        if index == 178:
            # for i, layer in enumerate(write_layer):
           # print('Capa', index, Net2.layers[index].__class__.__name__)
            a = outputs1[index] == outputs2[index]
            #out_1 = np.reshape(outputs1[index], (8, 1))
            # out_2 = np.reshape(outputs2[index], (8, 1))
            #outp_1.append(out_1)
            #outp_2.append(out_2)
            size_output = a.size
            output_true = np.sum(a)
            numero.append(index)
            capa.append(Net2.layers[index].__class__.__name__)
            list_output_true.append(output_true)
            list_size_output.append(size_output)
            amount_dif = size_output - output_true
            list_amount_dif.append(amount_dif)
            diff_nets_sum = np.sum(tf.math.abs(tf.math.subtract(outputs1[index], outputs2[index])))
            #diff_layer_softmax.append(diff_nets_sum)
            diff_nets_1_2 = tf.math.abs(tf.math.subtract(outputs1[index], outputs2[index]))
    return diff_nets_sum



acc_list = []
iterator = iter(test_dataset)
diff_layer_softmax = []
mean_diff_layer_softmaxnp=[]

activation_aging = [True]*29
# locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/locs_054')
# error_mask = load_obj('Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/error_mask_054')
#
#
# Net2 = GetNeuralNetworkModel('MobileNet', (224, 224, 3), 8, faulty_addresses=locs, masked_faults=error_mask,
#                              aging_active=True, word_size=word_size, frac_size=afrac_size,
#                              batch_size=testBatchSize)
# Net2.load_weights(wgt_dir).expect_partial()
# loss = tf.keras.losses.CategoricalCrossentropy()
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# WeightQuantization(model=Net2, frac_bits=wfrac_size, int_bits=wint_size)
# Net2.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
# loss, acc = Net2.evaluate(test_dataset)
# acc_list.append(acc)
# # list_ciclo.append(i)
# index = 1
# while index <= len(test_dataset):
#     image = next(iterator)[0]
#     print('index+++++++++++++++++++++++++++', index)
#     # print('imagen',index, image)
#     outputs1 = get_all_outputs(Net1, image)
#     # salidas del modelo sin fallas para la primer imagen del dataset de prueba
#     outputs2 = get_all_outputs(Net2, image)
#     # salidas del modelo con fallas para la primer imagen del dataset de prueba
#     diff_nets_sum = SofmaxDiffElements(outputs1, outputs2, acc_list)
#     diff_layer_softmax.append(diff_nets_sum)
#     index = index + 1
# df_diff_softmax_base = pd.DataFrame(diff_layer_softmax)
# mean_diff_layer_softmaxnp.append(np.mean(diff_layer_softmax))
# del diff_layer_softmax
#
# locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/ECCx4x1/ECCx/vc_707/locs_054')
# error_mask = load_obj('Data/Fault Characterization/variante_mask_vc_707/ECCx4x1/ECCx/vc_707/error_mask_054')
# iterator = iter(test_dataset)
# diff_layer_softmax = []
#
# Net2 = GetNeuralNetworkModel('MobileNet', (224, 224, 3), 8, faulty_addresses=locs, masked_faults=error_mask,
#                              aging_active=True, word_size=word_size, frac_size=afrac_size,
#                              batch_size=testBatchSize)
# Net2.load_weights(wgt_dir).expect_partial()
# loss = tf.keras.losses.CategoricalCrossentropy()
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# WeightQuantization(model=Net2, frac_bits=wfrac_size, int_bits=wint_size)
# Net2.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
# loss, acc = Net2.evaluate(test_dataset)
# acc_list.append(acc)
# # list_ciclo.append(i)
# index = 1
# while index <= len(test_dataset):
#     image = next(iterator)[0]
#     print('index+++++++++++++++++++++++++++', index)
#     # print('imagen',index, image)
#     outputs1 = get_all_outputs(Net1, image)
#     # salidas del modelo sin fallas para la primer imagen del dataset de prueba
#     outputs2 = get_all_outputs(Net2, image)
#     # salidas del modelo con fallas para la primer imagen del dataset de prueba
#     diff_nets = SofmaxDiffElements(outputs1, outputs2, acc_list)
#     diff_layer_softmax.append(diff_nets)
#     index = index + 1
# df_diff_softmax_IA_ECC = pd.DataFrame(diff_layer_softmax)
# mean_diff_layer_softmaxnp.append(np.mean(diff_layer_softmax))
# del diff_layer_softmax
#
# locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/ECC/ECC/vc_707/locs_054')
# error_mask = load_obj('Data/Fault Characterization/variante_mask_vc_707/ECC/ECC/vc_707/error_mask_054')
# # acc_list = []
# iterator = iter(test_dataset)
# diff_layer_softmax = []
#
# Net2 = GetNeuralNetworkModel('MobileNet', (224, 224, 3), 8, faulty_addresses=locs, masked_faults=error_mask,
#                              aging_active=True, word_size=word_size, frac_size=afrac_size,
#                              batch_size=testBatchSize)
# Net2.load_weights(wgt_dir).expect_partial()
# loss = tf.keras.losses.CategoricalCrossentropy()
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# WeightQuantization(model=Net2, frac_bits=wfrac_size, int_bits=wint_size)
# Net2.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
# loss, acc = Net2.evaluate(test_dataset)
# acc_list.append(acc)
# # list_ciclo.append(i)
# index = 1
# while index <= len(test_dataset):
#     image = next(iterator)[0]
#     print('index+++++++++++++++++++++++++++', index)
#     # print('imagen',index, image)
#     outputs1 = get_all_outputs(Net1, image)
#     # salidas del modelo sin fallas para la primer imagen del dataset de prueba
#     outputs2 = get_all_outputs(Net2, image)
#     # salidas del modelo con fallas para la primer imagen del dataset de prueba
#     diff_nets_sum = SofmaxDiffElements(outputs1, outputs2, acc_list)
#     diff_layer_softmax.append(diff_nets_sum)
#     index = index + 1
# df_diff_softmax_ECC = pd.DataFrame(diff_layer_softmax)
# mean_diff_layer_softmaxnp.append(np.mean(diff_layer_softmax))
# del diff_layer_softmax
#
#
#
# locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/mask_volteada_x/mask_volteada/vc_707/locs_054')
# error_mask = load_obj(
#     'Data/Fault Characterization/variante_mask_vc_707/mask_volteada_x/mask_volteada/vc_707/error_mask_054')
# # acc_list = []
# iterator = iter(test_dataset)
# diff_layer_softmax = []
#
# Net2 = GetNeuralNetworkModel('MobileNet', (224, 224, 3), 8, faulty_addresses=locs, masked_faults=error_mask,
#                              aging_active=True, word_size=word_size, frac_size=afrac_size,
#                              batch_size=testBatchSize)
# Net2.load_weights(wgt_dir).expect_partial()
# loss = tf.keras.losses.CategoricalCrossentropy()
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# WeightQuantization(model=Net2, frac_bits=wfrac_size, int_bits=wint_size)
# Net2.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
# loss, acc = Net2.evaluate(test_dataset)
# acc_list.append(acc)
# # list_ciclo.append(i)
# index = 1
# while index <= len(test_dataset):
#     image = next(iterator)[0]
#     print('index+++++++++++++++++++++++++++', index)
#     # print('imagen',index, image)
#     outputs1 = get_all_outputs(Net1, image)
#     # salidas del modelo sin fallas para la primer imagen del dataset de prueba
#     outputs2 = get_all_outputs(Net2, image)
#     # salidas del modelo con fallas para la primer imagen del dataset de prueba
#     diff_nets_sum = SofmaxDiffElements(outputs1, outputs2, acc_list)
#     diff_layer_softmax.append(diff_nets_sum)
#     index = index + 1
# df_diff_softmax_Flip = pd.DataFrame(diff_layer_softmax)
# mean_diff_layer_softmaxnp.append(np.mean(diff_layer_softmax))
# del diff_layer_softmax




error_mask = load_obj('Data/Fault Characterization/variante_mask_vc_707/mask_VBW_en_x/mask_VBW_base/vc_707/error_mask_054')
locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/mask_VBW_en_x/mask_VBW_base/vc_707/locs_054')
# acc_list = []
iterator = iter(test_dataset)
diff_layer_softmax = []
# mean_diff_layer_softmaxnp=[]

Net2 = GetNeuralNetworkModel('MobileNet', (224, 224, 3), 8, faulty_addresses=locs, masked_faults=error_mask,
                             aging_active=activation_aging, word_size=word_size, frac_size=afrac_size,
                             batch_size=testBatchSize)
Net2.load_weights(wgt_dir).expect_partial()
loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
WeightQuantization(model=Net2, frac_bits=wfrac_size, int_bits=wint_size)
Net2.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
loss, acc = Net2.evaluate(test_dataset)
acc_list.append(acc)
# list_ciclo.append(i)
index = 1
while index <= len(test_dataset):
    image = next(iterator)[0]
    print('index+++++++++++++++++++++++++++', index)
    # print('imagen',index, image)
    outputs1 = get_all_outputs(Net1, image)
    # salidas del modelo sin fallas para la primer imagen del dataset de prueba
    outputs2 = get_all_outputs(Net2, image)
    # salidas del modelo con fallas para la primer imagen del dataset de prueba
    diff_nets_sum = SofmaxDiffElements(outputs1, outputs2, acc_list)
    diff_layer_softmax.append(diff_nets_sum)
    index = index + 1

df_diff_softmax_Flip_Patch = pd.DataFrame(diff_layer_softmax)
mean_diff_layer_softmaxnp.append(np.mean(diff_layer_softmax))
del diff_layer_softmax

print('direrencias todas las redes',mean_diff_layer_softmaxnp)
df_diff_layer_softmax = pd.DataFrame(mean_diff_layer_softmaxnp)
df_acc_list = pd.DataFrame(acc_list)

buf_same_elemen = pd.concat([df_diff_softmax_Flip_Patch], axis=1, join='outer')
buf_same_elemen.columns = ['F + P']
acc_media_comprobacion = pd.concat([df_diff_layer_softmax,df_acc_list], axis=1, join='outer')
acc_media_comprobacion.columns = ['media_diff','ACC']
print(buf_same_elemen)
with pd.ExcelWriter('MobileNet/métricas/MobileNet_diff_softmax_f_p_vbw_16xx_prueba.xlsx') as writer:
    buf_same_elemen.to_excel(writer, sheet_name='softmax_exp', index=False)
    acc_media_comprobacion.to_excel(writer, sheet_name='acc_media_comprobacion', index=False)


# buf_same_elemen = pd.concat([df_diff_softmax_base,df_diff_softmax_IA_ECC,df_diff_softmax_ECC,df_diff_softmax_Flip,df_diff_softmax_Flip_Patch], axis=1, join='outer')
# buf_same_elemen.columns = ['Base','I-A ECC','ECC','Flip', 'F + P']
# acc_media_comprobacion = pd.concat([df_diff_layer_softmax,df_acc_list], axis=1, join='outer')
# acc_media_comprobacion.columns = ['media_diff','ACC']
# print(buf_same_elemen)
# with pd.ExcelWriter('MobileNet/métricas/MobileNet_diff_softmax_imagenes.xlsx') as writer:
#     buf_same_elemen.to_excel(writer, sheet_name='softmax_exp', index=False)
#     acc_media_comprobacion.to_excel(writer, sheet_name='acc_media_comprobacion', index=False)

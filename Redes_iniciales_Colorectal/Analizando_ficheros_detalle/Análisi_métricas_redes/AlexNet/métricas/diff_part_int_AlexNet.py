#!/usr/bin/env python
# coding: utf-8

# ## AlexNet

# In[1]:

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

import os
import pickle as pickle
import tensorflow as tf
import numpy as np
from Stats import WeightQuantization, IntroduceFaultsInWeights, GenerateFaultsList
# from Netsdiffelem import GetNeuralNetworkModel
from Nets_original import GetNeuralNetworkModel
#from Nets import GetNeuralNetworkModel
from Training import GetDatasets
from Simulation import get_all_outputs
from Simulation import buffer_simulation, save_obj, load_obj
from datetime import datetime
import pandas as pd
#from funciones import compilNet, same_elements

# In[2]:


# trainBatchSize = testBatchSize = 1
trainBatchSize = testBatchSize = 1
_, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (227, 227), 8, trainBatchSize, testBatchSize)

# In[3]:p

# Numero de bits para activaciones (a) y pesos (w)
word_size = 16
afrac_size = 11
aint_size = 4
wfrac_size = 11
wint_size = 4

# Tamaño del buffer de activaciones == al tamaño de la capa con mayor numero de activaciones (290400 pesos de 16 bits cada uno)

# Directorio de los pesos
cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'AlexNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir, 'Weights')

# In[4]:


activation_aging = [False] * 11

Net1 = GetNeuralNetworkModel('AlexNet', (227, 227, 3), 8, aging_active=activation_aging,
                             word_size=word_size, frac_size=afrac_size, batch_size=testBatchSize)
Net1.load_weights(wgt_dir).expect_partial()
loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
WeightQuantization(model=Net1, frac_bits=wfrac_size, int_bits=wint_size)
Net1.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
# pesos1 = Net1.get_weights()
loss_sf, acc_sf = Net1.evaluate(test_dataset)

# In[5]:

# a matriz que devuelve en true los elementos iguales entre el modelo con fallo y sin fallo
# size_output cantidad de elementos iguales entre el modelo con fallo y sin fallo
# amount_dif : cantidad de activaciones diferentes entre modelo con fallo y sin fallo
list_size_output = []
list_size_output = []
list_output_true = []
list_amount_dif = []
numero = []
capa = []
diff_nets_int=[]
diff_nets_int_abs=[]
diff_layer=[]


def diferenc_intpart(outputs1, outputs2, ciclo, acc_list):
    write_layer = [2, 8, 10, 16, 18, 24, 30, 36, 38, 44, 49, 53]

    for index in range(0, len(outputs2)):
        if index == write_layer[ciclo]:
            # for i, layer in enumerate(write_layer):
            print('Capa', index, Net2.layers[index].__class__.__name__)
            a = outputs1[index] == outputs2[index]
            size_output = a.size
            output_true = np.sum(a)
            numero.append(index)
            capa.append(Net2.layers[index].__class__.__name__)
            list_output_true.append(output_true)
            list_size_output.append(size_output)
            amount_dif = size_output - output_true
            list_amount_dif.append(amount_dif)
            diff_nets = np.sum(tf.math.abs(tf.math.subtract(outputs1[index], outputs2[index])))
            diff_layer.append(diff_nets)
            # np.trunc(outputs1[0])
            diff_netsint_abs = np.sum(tf.math.abs(tf.math.subtract(np.trunc(outputs1[index]), np.trunc(outputs2[index]))))
            diff_nets_int_abs.append(diff_netsint_abs)
            diff_netsint_signo = np.sum(tf.math.subtract(np.trunc(outputs1[index]), np.trunc(outputs2[index])))
            diff_nets_int.append(diff_netsint_signo)
    return (numero, capa, list_size_output, list_amount_dif, diff_layer, diff_nets_int_abs, diff_nets_int)


#


# In[7]:

def SofmaxDiffElements(outputs1, outputs2, acc_list):
    list_size_output = []
    list_output_true = []
    list_amount_dif = []
    numero = []
    capa = []



    for index in range(0, len(outputs2)):

        if index == 52:
            # for i, layer in enumerate(write_layer):
            print('Capa', index, Net2.layers[index].__class__.__name__)
            a = outputs1[index] == outputs2[index]
            size_output = a.size
            output_true = np.sum(a)
            numero.append(index)
            capa.append(Net2.layers[index].__class__.__name__)
            list_output_true.append(output_true)
            list_size_output.append(size_output)
            amount_dif = size_output - output_true
            list_amount_dif.append(amount_dif)
            #Si deseo sumar las diferencias para opneter soño un valor
            diff_nets_sum = np.sum(tf.math.abs(tf.math.subtract(outputs1[index], outputs2[index])))
            print('diff_nets_sum', diff_nets_sum)



    #return diff_nets_sum
    return  diff_nets_sum






locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/locs_054')
error_mask = load_obj('Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/error_mask_054')
acc_list = []
iterator = iter(test_dataset)
diff_layer_softmax = []
mean_diff_layer_softmaxnp=[]
diff_values_softmax = []
amount_imagen =[]

#with pd.ExcelWriter('AlexNet/AlexNet_prueba_base_all.xlsx') as writer:

Net2 = GetNeuralNetworkModel('AlexNet', (227, 227, 3), 8, faulty_addresses=locs, masked_faults=error_mask,
                             aging_active=True, word_size=word_size, frac_size=afrac_size,
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
# while index <= len(test_dataset):
while index <= 10:
    image = next(iterator)[0]
    print('index+++++++++++++++++++++++++++', index)
    # print('imagen',index, image)
    outputs1 = get_all_outputs(Net1, image)
    # salidas del modelo sin fallas para la primer imagen del dataset de prueba
    outputs2 = get_all_outputs(Net2, image)
    # salidas del modelo con fallas para la primer imagen del dataset de prueba
    diff_nets_sum = SofmaxDiffElements(outputs1, outputs2, acc_list)
    #amount_imagen.append(index)
    diff_layer_softmax.append(diff_nets_sum)
    # diff_values_image = SofmaxDiffElements(outputs1, outputs2, acc_list)

    # diff_values_softmax.append(diff_values_image)
    print('valores que quiero', diff_layer_softmax)
    print('tamaño resoltado final', (len(diff_layer_softmax)))
    index = index + 1

mean_diff_layer_softmaxnp.append(np.mean(diff_layer_softmax))
df_diff_softmax_base = pd.DataFrame(diff_layer_softmax)
#df_amount_imagen = pd.DataFrame(amount_imagen)
#buf_same_elemen = pd.concat([df_amount_imagen,df_diff_layer_softmax], axis=1, join='outer')
#buf_same_elemen.columns = ['df_experimentos','diff_layer_softmax']
print('media df_diff_softmax_base' ,mean_diff_layer_softmaxnp)

#with pd.ExcelWriter('AlexNet/métricas/AlexNet_diff_softmax_imagen_base.xlsx') as writer:
    #buf_same_elemen.to_excel(writer, sheet_name='base', index=False)
del diff_layer_softmax





locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/ECCx4x1/ECCx/vc_707/locs_054')
error_mask = load_obj('Data/Fault Characterization/variante_mask_vc_707/ECCx4x1/ECCx/vc_707/error_mask_054')
iterator = iter(test_dataset)
diff_layer_softmax = []
diff_values_softmax = []

Net2 = GetNeuralNetworkModel('AlexNet', (227, 227, 3), 8, faulty_addresses=locs, masked_faults=error_mask,
                             aging_active=True, word_size=word_size, frac_size=afrac_size,
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
while index <= 10:
    image = next(iterator)[0]
    print('index+++++++++++++++++++++++++++', index)
    # print('imagen',index, image)
    outputs1 = get_all_outputs(Net1, image)
    # salidas del modelo sin fallas para la primer imagen del dataset de prueba
    outputs2 = get_all_outputs(Net2, image)
    # salidas del modelo con fallas para la primer imagen del dataset de prueba
    diff_nets_sum=SofmaxDiffElements(outputs1, outputs2, acc_list)
    diff_layer_softmax.append(diff_nets_sum)

    index = index + 1
df_diff_softmax_IA_ECC = pd.DataFrame(diff_layer_softmax)
mean_diff_layer_softmaxnp.append(np.mean(diff_layer_softmax))
print('media df_diff_softmax_IA_ECC' ,mean_diff_layer_softmaxnp)

del diff_layer_softmax


locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/ECC/ECC/vc_707/locs_054')
error_mask = load_obj('Data/Fault Characterization/variante_mask_vc_707/ECC/ECC/vc_707/error_mask_054')
# acc_list = []
iterator = iter(test_dataset)
diff_layer_softmax = []

Net2 = GetNeuralNetworkModel('AlexNet', (227, 227, 3), 8, faulty_addresses=locs, masked_faults=error_mask,
                             aging_active=True, word_size=word_size, frac_size=afrac_size,
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
while index <= 10:
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
mean_diff_layer_softmaxnp.append(np.mean(diff_layer_softmax))
df_diff_softmax_ECC = pd.DataFrame(diff_layer_softmax)
del diff_layer_softmax
locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/mask_volteada_x/mask_volteada/vc_707/locs_054')
error_mask = load_obj(    'Data/Fault Characterization/variante_mask_vc_707/mask_volteada_x/mask_volteada/vc_707/error_mask_054')
# acc_list = []
iterator = iter(test_dataset)
diff_layer_softmax = []

Net2 = GetNeuralNetworkModel('AlexNet', (227, 227, 3), 8, faulty_addresses=locs, masked_faults=error_mask,
                             aging_active=True, word_size=word_size, frac_size=afrac_size,
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
while index <= 10:
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
mean_diff_layer_softmaxnp.append(np.mean(diff_layer_softmax))
print('media df_diff_softmax_Flip' ,mean_diff_layer_softmaxnp)
df_diff_softmax_Flip = pd.DataFrame(diff_layer_softmax)

del diff_layer_softmax



locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/flip_patch/byte_x/vc_707/locs_054')
error_mask = load_obj('Data/Fault Characterization/variante_mask_vc_707/flip_patch/byte_x/vc_707/error_mask_054')
# acc_list = []
iterator = iter(test_dataset)
diff_layer_softmax = []
# mean_diff_layer_softmaxnp=[]


Net2 = GetNeuralNetworkModel('AlexNet', (227, 227, 3), 8, faulty_addresses=locs, masked_faults=error_mask,
                             aging_active=True, word_size=word_size, frac_size=afrac_size,
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
while index <= 10:
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
mean_diff_layer_softmaxnp.append(np.mean(diff_layer_softmax))
df_diff_softmax_Flip_Patch = pd.DataFrame(diff_layer_softmax)

del diff_layer_softmax

print('direrencias todas las redes',mean_diff_layer_softmaxnp)
df_diff_layer_softmax = pd.DataFrame(mean_diff_layer_softmaxnp)
df_acc_list = pd.DataFrame(acc_list)

buf_same_elemen = pd.concat([df_diff_softmax_base,df_diff_softmax_IA_ECC,df_diff_softmax_ECC,df_diff_softmax_Flip,df_diff_softmax_Flip_Patch], axis=1, join='outer')
buf_same_elemen.columns = ['Base','I-A ECC','ECC','Flip', 'F + P']
acc_media_comprobacion = pd.concat([df_diff_layer_softmax,df_acc_list], axis=1, join='outer')
acc_media_comprobacion.columns = ['media_diff','ACC']
print(buf_same_elemen)
with pd.ExcelWriter('AlexNet/métricas/AlexNet_diff_softmax_prueba.xlsx') as writer:
    buf_same_elemen.to_excel(writer, sheet_name='softmax_exp', index=False)
    acc_media_comprobacion.to_excel(writer, sheet_name='acc_media_comprobacion', index=False)
    # df_diff_layer_softmax = pd.DataFrame(mean_diff_layer_softmaxnp)
    # df_acc_list = pd.DataFrame(acc_list)
    # experimentos = ['base', 'ECC_base_x4', 'ECC', '1bytex', 'base_Volt']
    # df_experimentos = pd.DataFrame(experimentos)
    # buf_same_elemen = pd.concat([df_experimentos,df_diff_layer_softmax, df_acc_list], axis=1, join='outer')
    # buf_same_elemen.columns = ['df_experimentos','diff_layer_softmax', 'df_acc_list']
    # print(buf_same_elemen)
    # buf_same_elemen.to_excel(writer, sheet_name='salidas', index=False)
    # writer.save()


#Calcular la diferencia entre la part eentera del modelo con fallos y sin fallos
# Cargar_errores = True
#
# if Cargar_errores:
#     locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask/vc_707/locs_054')
#     error_mask = load_obj('Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask/vc_707/error_mask_054')
#
#
# activation_aging = np.array([False] * 11)
# acc_list = []
# list_ciclo = []
#
# with pd.ExcelWriter('AlexNet/AlexNet_ratio_part_int_base.xlsx') as writer:
#     for i, valor in enumerate(activation_aging):
#         ciclo = i
#         activation_aging[i] = True
#         activation_aging[i - 1] = False
#         print(activation_aging)
#         # activation_aging = [False,False,False,False,True,False,False,False,False,False,False]
#         # activation_aging= False
#         Net2 = GetNeuralNetworkModel('AlexNet', (227, 227, 3), 8, faulty_addresses=locs, masked_faults=error_mask,
#                                      aging_active=activation_aging, word_size=word_size, frac_size=afrac_size,
#                                      batch_size=testBatchSize)
#         Net2.load_weights(wgt_dir).expect_partial()
#         loss = tf.keras.losses.CategoricalCrossentropy()
#         optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
#         WeightQuantization(model=Net2, frac_bits=wfrac_size, int_bits=wint_size)
#         Net2.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
#         loss, acc = Net2.evaluate(test_dataset)
#         acc_list.append(acc)
#         # list_ciclo.append(i)
#
#         X = [x for x, y in test_dataset]
#         # salidas del modelo sin fallas para la primer imagen del dataset de prueba
#         outputs1 = get_all_outputs(Net1, X[0])
#         # salidas del modelo con fallas para la primer imagen del dataset de prueba
#         outputs2 = get_all_outputs(Net2, X[0])
#         # print(outputs1)
#         # print(outputs2)
#         # print('list_ciclo',list_ciclo)
#         # same_elements(outputs1,outputs2,list_ciclo,acc_list)
#         diferenc_intpart(outputs1, outputs2, ciclo, acc_list)
#     if ciclo == len(activation_aging) - 1:
#         last_layer = ciclo + 1
#         diferenc_intpart(outputs1, outputs2, last_layer, 0)
#
#     df_numero = pd.DataFrame(numero)
#     df_capa = pd.DataFrame(capa)
#     df_acc = pd.DataFrame(acc_list)
#     df_list_size_output = pd.DataFrame(list_size_output)
#     df_list_output_diff = pd.DataFrame(list_amount_dif)
#     df_diff_nets = pd.DataFrame(diff_layer)
#     df_diff_netsint_abso=pd.DataFrame(diff_nets_int_abs)
#     df_diff_netsint_signo=pd.DataFrame(diff_nets_int)
#
#     buf_same_elemen = pd.concat([df_numero, df_capa, df_list_size_output, df_acc, df_list_output_diff, df_diff_nets, df_diff_netsint_abso,
#          df_diff_netsint_signo], axis=1, join='outer')
#     buf_same_elemen.columns = ['Num', 'Capa', 'T_actv', 'Acc', '#_Act_diff', 'diff_nets', 'diff_netsint_abso', 'diff_netsint_signo']
#     buf_same_elemen.to_excel(writer, sheet_name='datos1', startcol=2, index=False)
#     writer.save()

print('Ejecución  completada: ', datetime.now().strftime("%H:%M:%S"))

# In[12]:


acc_list
acc_list_np = np.asarray(acc_list)
print(acc_list_np)

# In[14]:


# In[8]:


# In[ ]:





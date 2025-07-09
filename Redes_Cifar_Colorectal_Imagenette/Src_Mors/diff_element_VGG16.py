#!/usr/bin/env python
# coding: utf-8

# ## VGG16

# In[1]:

#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

import os
import pickle as pickle
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from Stats_Cifar_ import WeightQuantization, IntroduceFaultsInWeights, GenerateFaultsList
#from Nets_original import GetNeuralNetworkModel
from Nets_mobile import GetNeuralNetworkModel
from Training import GetDatasets
from Simulation import get_all_outputs
from Simulation import buffer_simulation, save_obj, load_obj
from datetime import datetime
import pandas as pd
from keras.optimizers import Adam
from funciones import compilNet, same_elements


# In[2]:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(train_ds, validation_ds, test_ds), info = tfds.load('cifar10',
                                    split=["train", "test[:35%]", "test[35%:]"],
                                    as_supervised = True,
                                    with_info=True,
                                    shuffle_files= True)


size = (32, 32)
#batch_size = 128
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


# Numero de bits para activaciones (a) y pesos (w)
word_size  = 16
afrac_size = 13
aint_size  = 2
wfrac_size = 15
wint_size  = 0

# Tamaño del buffer de activaciones == al tamaño de la capa con mayor numero de activaciones (290400 pesos de 16 bits cada uno)

# Directorio de los pesos
wgt_dir= ('../Trained_Weights_cifar/VGG16/weights.data')





activation_aging = [False]*21


#Acá la creamos, notese que como no se introduciran fallos en activaciones no es necesario pasar locs ni masks
Net1 = GetNeuralNetworkModel('VGG16', (32,32,3), 10, aging_active=activation_aging,
                             word_size=word_size, frac_size=afrac_size, batch_size = 1)
Net1.load_weights(wgt_dir).expect_partial()
loss = 'sparse_categorical_crossentropy'
optimizer = Adam()
WeightQuantization(model = Net1, frac_bits = wfrac_size, int_bits = wint_size)
Net1.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
loss_sf,acc_sf =Net1.evaluate(test_ds)


print('primera parte vencida')


# In[5]:

# def same_elements(outputs1, outputs2, ciclo):
#     list_size_output = []
#     list_output_true = []
#     list_amount_dif = []
#     list_ratio = []
#     list_ratio_zero = []
#     list_total_zero = []
#     numero = []
#     capa = []
#
#     for index in range(0, len(outputs2)):
#         # print('Capa',index,Net2.layers[index].__class__.__name__)
#         a = outputs1[index] == outputs2[index]
#         size_output = a.size
#         output_true = np.sum(a)
#         numero.append(index)
#         capa.append(Net2.layers[index].__class__.__name__)
#         list_output_true.append(output_true)
#         list_size_output.append(size_output)
#         amount_dif = size_output - output_true
#         list_amount_dif.append(amount_dif)
#         ratio = ((amount_dif * 100) / size_output)
#         list_ratio.append(ratio)
#         non_zero = (np.count_nonzero(outputs2[index]))
#         total_zero = size_output - non_zero
#         ratio_zero = ((total_zero * 100) / size_output)
#         list_ratio_zero.append(ratio_zero)
#         list_total_zero.append(total_zero)
#
#         df_numero = pd.DataFrame(numero)
#         df_capa = pd.DataFrame(capa)
#         df_list_size_output = pd.DataFrame(list_size_output)
#         df_list_output_diff = pd.DataFrame(list_amount_dif)
#         df_list_ratio = pd.DataFrame(list_ratio)
#         df_list_elem_zero = pd.DataFrame(list_total_zero)
#         df_list_ratio_zero = pd.DataFrame(list_ratio_zero)
#
#         buf_same_elemen = pd.concat(
#             [df_numero, df_capa, df_list_size_output, df_list_output_diff, df_list_ratio, df_list_elem_zero,
#              df_list_ratio_zero], axis=1, join='outer')
#         buf_same_elemen.columns = ['num', 'capa', 'Total_elements_layer', 'diff_elements', 'Ratio', 'amount_zero',
#                                    'Ratio']
#         buf_same_elemen.to_excel(writer, sheet_name='ratio_' + str(ciclo), index=False)


# In[7]:
#a matriz que devuelve en true los elementos iguales entre el modelo con fallo y sin fallo
#size_output cantidad de elementos iguales entre el modelo con fallo y sin fallo
#amount_dif : cantidad de activaciones diferentes entre modelo con fallo y sin fallo
#
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
    write_layer = [2,6,10,12,16,20,22,26,30,34,36,40,44,48,50,54,58,62,64,69,73,77,79]

    for index in range(0, len(outputs2)):
        if index == write_layer[ciclo]:
            # for i, layer in enumerate(write_layer):
            print('Capa', index, Net2.layers[index].__class__.__name__)
            a = outputs1[index] == outputs2[index]
            size_output = a.size
            print('size_output',size_output)
            output_true = np.sum(a)
            print('output_true',output_true)
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


#Decidir si cargar errores de un archivo locs y error_mask o generarlos aleatoriamente
Cargar_errores = True


if Cargar_errores:
    error_mask = load_obj('Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/error_mask_054')
    locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/locs_054')

""" else:
    locs = load_obj('Data/Fault Characterization/error_mask_x_10/error_mask_707/locs_0_54_buffer_act')
    numero_bits_con_fallo = len(locs)
    bits_con_fallo = np.random.randint(0,1,numero_bits_con_fallo)
    mbuffer = np.array(['0']*(abuffer_size-numero_bits_con_fallo))
    mbuffer = np.concatenate([mbuffer,bits_con_fallo])
    #Convertirlo en vectores de 16 elementos
    mbuffer = np.reshape(mbuffer,(-1,word_size))
    mbuffer = ["".join(i) for i in mbuffer]
    error_mask=mbuffer[0:numero_bits_con_fallo]
print('mostrando las 5 primeras direcciones con fallos')
print('direcciones:',locs[0:5])
print('mascara de fallos:',error_mask[0:5])
print(len(error_mask))

 """


activation_aging = np.array([False]*21)
acc_list=[]
list_ciclo=[]

#with pd.ExcelWriter('Experimentos/Cifar/VGG16/nume_actv_diffe_VGG16_cifar_errormask_x.xlsx') as writer:
    


for i, valor in enumerate(activation_aging):
    ciclo=i
    activation_aging[i]=True
    activation_aging[i-1]=False
    print (activation_aging)

    Net2 = GetNeuralNetworkModel('VGG16', (32,32,3), 10, faulty_addresses=locs, masked_faults=error_mask,  aging_active=activation_aging, word_size=word_size, frac_size=afrac_size,
                                 batch_size = 1)
    Net2.load_weights(wgt_dir).expect_partial()
    loss = 'sparse_categorical_crossentropy'
    optimizer = Adam()
    Net2.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
    WeightQuantization(model=Net2, frac_bits=wfrac_size, int_bits=wint_size)
    loss,acc  = Net2.evaluate(test_ds)
    acc_list.append(acc)
    #list_ciclo.append(i)
        
    X = [x for x,y in test_ds]
    #salidas del modelo sin fallas para la primer imagen del dataset de prueba
    outputs1= get_all_outputs(Net1,X[0])
    #salidas del modelo con fallas para la primer imagen del dataset de prueba
    outputs2 = get_all_outputs(Net2,X[0])
    #print(outputs1)
    #print(outputs2)
    #print('list_ciclo',list_ciclo)
    #same_elements(outputs1,outputs2,list_ciclo,acc_list)
    same_elements(outputs1, outputs2, ciclo, acc_list)
if ciclo==len(activation_aging)-1:
    last_layer=ciclo+1
    same_elements(outputs1, outputs2, last_layer, 0)
df_numero = pd.DataFrame(numero)
df_capa = pd.DataFrame(capa)
df_acc = pd.DataFrame(acc_list)
df_list_size_output = pd.DataFrame(list_size_output)
df_list_output_diff = pd.DataFrame(list_amount_dif)
df_list_ratio = pd.DataFrame(list_ratio)
df_list_elem_zero = pd.DataFrame(list_total_zero)
df_list_ratio_zero = pd.DataFrame(list_ratio_zero)

buf_same_elemen = pd.concat([df_numero, df_capa, df_list_size_output, df_acc, df_list_output_diff, df_list_ratio, df_list_elem_zero,
                      df_list_ratio_zero], axis=1, join='outer')
buf_same_elemen.columns = ['Num', 'Capa', 'T_actv', 'Acc', '#_Act_diff', 'perc', 'Act_0', 'perc']
buf_same_elemen.to_excel('new_nume_actv_diffe_VGG16_cifar_errormask_x.xlsx', sheet_name='datos1',  index=False)


        
    
        
print('Ejecución  completada: ', datetime.now().strftime("%H:%M:%S"))   
        
        
    
    


# In[12]:


acc_list


# In[14]:


acc_list_np = np.asarray(acc_list)
print(acc_list_np)
VGG16 = pd.DataFrame(acc_list_np)
#VGG16.columns = ['acc']
#VGG16.to_excel('VGG16', sheet_name='acc', index=False)



# ## VGG16

# In[8]:




# In[ ]:





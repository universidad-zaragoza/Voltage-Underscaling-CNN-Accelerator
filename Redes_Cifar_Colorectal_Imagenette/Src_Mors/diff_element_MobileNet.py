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
from Nets import GetNeuralNetworkModel
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


#Acá la creamos, notese que como no se introduciran fallos en activaciones no es necesario pasar locs ni masks
Net1 = GetNeuralNetworkModel('MobileNet', (224,224,3), 8, aging_active=activation_aging,
                             word_size=word_size, frac_size=afrac_size, batch_size = testBatchSize)
Net1.load_weights(wgt_dir).expect_partial()
WeightQuantization(model = Net1, frac_bits = wfrac_size, int_bits = wint_size)
loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
Net1.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
#pesos1 = Net1.get_weights()
loss_sf,acc_sf =Net1.evaluate(test_dataset)


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


#Decidir si cargar errores de un archivo locs y error_mask o generarlos aleatoriamente
Cargar_errores = True


if Cargar_errores:
    """ locs = load_obj('Data/Fault Characterization/error_mask y locs_buffer_act_vc-707/locs_0_54')
    error_mask = load_obj('Data/Fault Characterization/error_mask y locs_buffer_act_vc-707/error_mask_0_54') """
    locs = load_obj('Data/Fault Characterization/error_mask_x_100/error_mask_707/locs_0_54_buffer_act')
    error_mask = load_obj('Data/Fault Characterization/error_mask_x_100/error_mask_707/error_mask_0_54_buffer_act')
    """ locs  = load_obj('Data/Fault Characterization/error_mask y locs_buffer_pesos_vc-707/locs_0_54_buffer_pesos')
    error_mask = load_obj('Data/Fault Characterization/error_mask y locs_buffer_pesos_vc-707/error_mask_0_54_buffer_pesos') """
else:
    locs = load_obj('Data/Fault Characterization/error_mask_x_10/error_mask_707/locs_0_54_buffer_act')
    numero_bits_con_fallo = len(locs)
    bits_con_fallo = np.random.randint(0,1,numero_bits_con_fallo)
    mbuffer = np.array(['0']*(abuffer_size-numero_bits_con_fallo))
    mbuffer = np.concatenate([mbuffer,bits_con_fallo])
    #Convertirlo en vectores de 16 elementos
    mbuffer = np.reshape(mbuffer,(-1,word_size))
    mbuffer = ["".join(i) for i in mbuffer]
    error_mask=mbuffer[0:numero_bits_con_fallo]
    #more_posiction_errors=len(error_mask)-len(locs)
print('mostrando las 5 primeras direcciones con fallos')
print('direcciones:',locs[0:5])
print('mascara de fallos:',error_mask[0:5])
print(len(error_mask))




activation_aging = np.array([False]*29)
acc_list=[]
list_ciclo=[]

with pd.ExcelWriter('MobileNet/ratio_element_diff_MobileNet_error_x100.xlsx') as writer:
    


    for i, valor in enumerate(activation_aging):
        ciclo=i
        activation_aging[i]=True 
        activation_aging[i-1]=False    
        print (activation_aging)
    #activation_aging = [False,False,False,False,True,False,False,False,False,False,False]
    #activation_aging= False
        Net2 = GetNeuralNetworkModel('MobileNet', (224,224,3), 8, faulty_addresses=locs, masked_faults=error_mask,
                                 aging_active=activation_aging, word_size=word_size, frac_size=afrac_size, 
                                 batch_size = testBatchSize)
        Net2.load_weights(wgt_dir).expect_partial()
        loss = tf.keras.losses.CategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        Net2.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
        WeightQuantization(model=Net2, frac_bits=wfrac_size, int_bits=wint_size)
        loss,acc  = Net2.evaluate(test_dataset)
        acc_list.append(acc)
        #list_ciclo.append(i)
        
        X = [x for x,y in test_dataset]
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
    buf_same_elemen.columns = ['Num', 'Capa', 'T_actv', 'Acc', 'diff_actv', 'perc', 'Act_0', 'perc']
    buf_same_elemen.to_excel(writer, sheet_name='datos1', startcol=2, index=False)
    writer.save()
writer.close        

        
    
        
print('Ejecución  completada: ', datetime.now().strftime("%H:%M:%S"))   
        
        
    
    


# In[12]:


acc_list


# In[14]:





# ## MobileNet

# In[8]:




# In[ ]:





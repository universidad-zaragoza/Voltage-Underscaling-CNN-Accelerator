#!/usr/bin/env python
# coding: utf-8

# ## VGG16

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


trainBatchSize = testBatchSize = 10
_, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)


# In[3]:


# Numero de bits para activaciones (a) y pesos (w)
word_size  = 16
afrac_size = 12
aint_size  = 3
wfrac_size = 13
wint_size  = 2

# Tamaño del buffer de activaciones == al tamaño de la capa con mayor numero de activaciones (290400 pesos de 16 bits cada uno)
abuffer_size = 16777216
# Directorio de los pesos
cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'DenseNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir,'Weights')


# In[4]:


activation_aging = [False]*188


#Acá la creamos, notese que como no se introduciran fallos en activaciones no es necesario pasar locs ni masks
Net1 = GetNeuralNetworkModel('DenseNet', (224,224,3), 8, aging_active=activation_aging,
                             word_size=word_size, frac_size=afrac_size, batch_size = testBatchSize)
Net1.load_weights(wgt_dir).expect_partial()
loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
WeightQuantization(model = Net1, frac_bits = wfrac_size, int_bits = wint_size)
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

    write_layer = [2,9,11,15,21,(23,11),28,34,(36,24),41,47,(49,37),54,60,(62,50),67,73,(75,63),80,86,(88,76),93,96,98,102,108,(110,98),
     115,121,(123,111),128,134,(136,124),141,147,(149,137),154,160,(162,150),167,173,(175,163),180,186,(188,176),
     193,199,(201,189),206,212,(214,202),219,225,(227,215),232,238,(240,228),245,251,(253,241),258,261,263,267,273,(275,263),
     280,286,(288,276),293,299,(301,289),306,312,(314,302),319,325,(327,315),332,338,(340,328),345,351,(353,341),
     358,364,(366,353),371,377,(379,367),384,390,(392,380),397,403,(405,393),410,416,(418,406),423,429,(431,419),
     436,442,(444,432),449,455,(457,445),462,468,(470,458),475,481,(483,471),488,494,(496,484),501,507,(509,497),
     514,520,(522,510),527,533,(535,523),540,546,(548,536),553,559,(561,549),566,572,(574,562),579,582,584,588,594,(596,584),
     601,607,(609,597),614,620,(622,610),627,633,(635,623),640,646,(648,636),653,659,(661,649),666,672,(674,662),
     679,685,(687,675),692,698,(700,688),705,711,(713,701),718,724,(726,714),731,737,(739,727),744,750,(752,740),
     757,763,(765,753),770,776,(778,766),783,789,(791,779),796,799,803]

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
    error_mask = load_obj('Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/error_mask_054')
    locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/locs_054')
    
""" else:
    locs = load_obj('Data/Fault Characterization/error_mask y locs_buffer_act_vc-707/locs_0_54')
    numero_bits_con_fallo = len(locs)
    bits_con_fallo = np.random.randint(0,2,numero_bits_con_fallo)
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



activation_aging = np.array([False]*188)
acc_list=[]
list_ciclo=[]

with pd.ExcelWriter('DenseNet/ratio_element_diff_DenseNet_more_errorX_100.xlsx') as writer:
    


    for i, valor in enumerate(activation_aging):
        ciclo=i
        activation_aging[i]=True 
        activation_aging[i-1]=False    
        print (activation_aging)
    #activation_aging = [False,False,False,False,True,False,False,False,False,False,False]
    #activation_aging= False
        Net2 = GetNeuralNetworkModel('DenseNet', (224,224,3), 8, faulty_addresses=locs, masked_faults=error_mask,
                                 aging_active=activation_aging, word_size=word_size, frac_size=afrac_size, 
                                 batch_size = testBatchSize)
        Net2.load_weights(wgt_dir).expect_partial()
        loss = tf.keras.losses.CategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        Net2.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
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
    buf_same_elemen.columns = ['Num', 'Capa', 'T_actv', 'Acc', 'dif_actv', 'perc', 'Act_0', 'perc']
    buf_same_elemen.to_excel(writer, sheet_name='datos1', startcol=2, index=False)
    writer.save()
writer.close        

        
    
        
print('Ejecución  completada: ', datetime.now().strftime("%H:%M:%S"))   
        
        
    
    


# In[12]:


acc_list


# In[14]:


# acc_list_np = np.asarray(acc_list)
# print(acc_list_np)
# DenseNet = pd.DataFrame(acc_list_np)
#DenseNet.columns = ['acc']
#DenseNet.to_excel('VGG16', sheet_name='acc', index=False)



# ## VGG16

# In[8]:




# In[ ]:





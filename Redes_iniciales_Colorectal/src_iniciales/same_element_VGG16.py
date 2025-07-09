#!/usr/bin/env python
# coding: utf-8

# In[5]:

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


trainBatchSize = testBatchSize = 16
_, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (227, 227), 8, trainBatchSize, testBatchSize)


# In[3]:


# Numero de bits para activaciones (a) y pesos (w)
word_size  = 16
afrac_size = 11  
aint_size  = 4
wfrac_size = 11
wint_size  = 4

# Tamaño del buffer de activaciones == al tamaño de la capa con mayor numero de activaciones (290400 pesos de 16 bits cada uno)
abuffer_size = 16777216
# Directorio de los pesos
cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'VGG16')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir,'Weights')


# In[6]:


activation_aging = [False]*21
acc_list=[]
list_ciclo=[]


#Acá la creamos, notese que como no se introduciran fallos en activaciones no es necesario pasar locs ni masks
Net1 = GetNeuralNetworkModel('VGG16', (227,227,3), 8, aging_active=activation_aging,
                             word_size=word_size, frac_size=afrac_size, batch_size = testBatchSize)
Net1.load_weights(wgt_dir).expect_partial()
loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
#WeightQuantization(model = Net1, frac_bits = wfrac_size, int_bits = wint_size)
Net1.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
#pesos1 = Net1.get_weights()
loss_sf,acc_sf =Net1.evaluate(test_dataset)


# In[7]:


def same_elements(outputs1,outputs2,ciclo):
    
    list_size_output=[]
    list_output_true=[]
    list_ratio=[]
    
    
    
    for index in range(0,len(outputs2)):
        
        print('Capa',index,Net2.layers[index].__class__.__name__)
        a=outputs1[index]== outputs2[index]
        size_output=a.size
        output_true=np.sum(a)
        list_output_true.append(output_true)
        list_size_output.append(size_output)
        amount_dif=size_output-output_true
        ratio=(output_true*100)/size_output
        list_ratio.append(ratio)
        df_list_size_output=pd.DataFrame(list_size_output)
        df_list_output_true=pd.DataFrame(list_output_true)
        df_list_ratio=pd.DataFrame(list_ratio)
        df_list_capas=pd.DataFrame(list_ratio)
        buf_same_elemen = pd.concat([df_list_size_output,df_list_output_true, df_list_ratio], axis=1, join='outer')
        buf_same_elemen.columns = ['Total_elements_layer', 'Same_elements', 'Ratio']
        buf_same_elemen.to_excel(writer, sheet_name='ratio_'+ str(ciclo), index=False)
  


# In[8]:






#Decidir si cargar errores de un archivo locs y error_mask o generarlos aleatoriamente
Cargar_errores = True


if Cargar_errores:
    locs  = load_obj('Data/Fault Characterization/locs')
    error_mask = load_obj('Data/Fault Characterization/error_mask')
else:
    numero_bits_con_fallo = 1000
    bits_con_fallo = np.random.randint(0,2,numero_bits_con_fallo)
    #crear una mascara (m) del buffer de pesos donde x: bit sin fallo, 0: bit con fallo en 0, 1: bit con fallo en 1.
    #si quieres introducir fallos en activaciones en lugar de los pesos, simplemente cambia wbuffer_size por abuffer_size.
    mbuffer = np.array(['x']*(abuffer_size-numero_bits_con_fallo))
    mbuffer = np.concatenate([mbuffer,bits_con_fallo])
    #distribuir los errores aleatoriamente en la mascara del buffer
    np.random.shuffle(mbuffer)
    #organizar la mascara del buffer por direcciones
    mbuffer = np.reshape(mbuffer,(-1,word_size))
    mbuffer = ["".join(i) for i in mbuffer]
    #filtrar dejando solo las direcciones con error
    locs  = [x for x,y in enumerate(mbuffer) if y.count('x') < 16]
    masks = [y for x,y in enumerate(mbuffer) if y.count('x') < 16] 
print('mostrando las 5 primeras direcciones con fallos')
print('direcciones:',locs[0:5])
print('mascara de fallos:',error_mask[0:5])
print(len(error_mask))


trainBatchSize = testBatchSize = 16
_, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (227, 227), 8, trainBatchSize, testBatchSize)






with pd.ExcelWriter('VGG16/ratio_element_igual.xlsx') as writer:
    


    for i, valor in enumerate(activation_aging):
        ciclo=i
        activation_aging[i]=True 
        activation_aging[i-1]=False    
        print (activation_aging)
    #activation_aging = [False,False,False,False,True,False,False,False,False,False,False]
    #activation_aging= False
        Net2 = GetNeuralNetworkModel('VGG16', (227,227,3), 8, faulty_addresses=locs, masked_faults=error_mask,
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
        print(acc_list)
        same_elements(outputs1,outputs2,ciclo)
writer.close        

        
    
        
print('Ejecución  completada: ', datetime.now().strftime("%H:%M:%S"))   
        
        
    
    


# In[ ]:





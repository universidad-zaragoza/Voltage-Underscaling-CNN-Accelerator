#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Training import GetDatasets
from Nets  import GetNeuralNetworkModel
from Stats import WeightQuantization, ActivationStats, CheckAccuracyAndLoss, QuantizationEffect, GetReadAndWrites
from Simulation import buffer_simulation, save_obj, load_obj

tf.random.set_seed(1234)
np.random.seed(1234)


# # 1) Training

# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; a) Get Dataset

# Primero Obtendremos el dataset colorectal, especificando la distribucion de entrenamiento/validacion y testing (80%,5%,15% para este caso); las dimensiones de entrada de la imagen (227,227), el numero de clases (8) y el tamaño de los batches. 
# Nota: Pueden aparecer un monton de mensajes de tensorflow durante la primera ejecucion, no impactan en nada

# In[2]:


train_batch_size = test_batch_size = 32
train_set,valid_set,test_set = GetDatasets('colorectal_histology',(80,5,15),(227,227), 8, train_batch_size, test_batch_size)


# El resultado son datasets iterables como el siguiente:

# In[3]:


train_set


# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; b) Get Model

# Luego creamos la red, en principio no activaremos ni la cuantizacion ni el efecto de envejecimiento

# In[4]:


AlexNet   = GetNeuralNetworkModel('AlexNet',(227,227,3),8, quantization = False, aging_active=False)
loss      = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
AlexNet.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])


# con el metodo summary() se puede ver los detalles de la red capa por capa, las capas Lambda emplean la cuantizacion y envejecimiento, las cuales estan presentes pero no activas cuando estas opciones estan desactivadas

# In[17]:


AlexNet.summary()


# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; c) Training

# Si se utilizan pesos ya entrenados el entrenamiento se puede omitir

# In[18]:


# Early Stopping
# --------------
#earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
#    
#AlexNet.fit(x=train_set,epochs=100,
#            steps_per_epoch  =int(np.ceil(train_size / train_batch_size)),
#            validation_data  =valid_set,
#            validation_steps =int(np.ceil(valid_size/ train_batch_size)), 
#            callbacks=[earlyStop])


# a continuacion cargamos o guardamos los pesos en la ruta especificada usando el metodo load_weights/save_weights

# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; d) Load/Save Weigths

# In[19]:


cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'AlexNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir,'Weights')
AlexNet.load_weights(wgt_dir)


# podemos evaluar el accuracy y loss de la red

# In[8]:


(OrigLoss,OrigAcc) = AlexNet.evaluate(test_set)


# # 2) Stats

# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Write/Read Stats

# primero identificamos (manualmente) las capas procesadadas(Convoluciones, Full conectadas y Pooling) junto a las capas que contienen los resultados que se escribiran en el buffer (capas luego de la funcion de activacion y/o Normalizacion)

# In[20]:


for index,layer in enumerate(AlexNet.layers):
    print(index,layer.name)
print('Las capas 0,3,9,11,17,19,25,31,37,40,45 y 50  contienen la informacion para su procesamiento')
print('Las capas 2,8,10,16,18,24,30,36,38,44,49 y 53 contienen las activaciones que son escritas en memoria')


# con el siguiente bloque obtenemos el numero de lecturas y escrituras por posicion de memoria tanto usando la estrategia de CNN Gating o sin usarla

# In[21]:


num_address  = 290400  # Tamaño del buffer, igual a la capa mas grande de la red (este caso), o un tamaño fijo pre-establecido.
Indices      = [0,3,9,11,17,19,25,31,37,40,45,50] #Capas con la informacion de procesamiento 
samples      = 10 #Numero de imagenes
# Sin Power Gating:
Data         = GetReadAndWrites(AlexNet,Indices,num_address,samples,CNN_gating=False)
stats        = {'Lecturas': Data['Reads'],'Escrituras': Data['Writes']}
Baseline_Acceses   = pd.DataFrame(stats).reset_index(drop=False)
# Con Power Gating
Data     = GetReadAndWrites(AlexNet,Indices,num_address,samples,CNN_gating=True)
stats    = {'Lecturas': Data['Reads'],'Escrituras': Data['Writes']}
CNN_gating_Acceses = pd.DataFrame(stats).reset_index(drop=False)
#save_obj(Baseline_Acceses,'Data/Acceses/AlexNet/Colorectal Dataset/Baseline')
#save_obj(CNN_gating_Acceses,'Data/Acceses/AlexNet/CNN_gating_Adj')


# Como resultado tenemos los siguientes dataframes

# In[ ]:


Baseline_Acceses


# In[ ]:


CNN_gating_Acceses


# # 3) Quantization 

# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; a) Number of bits analysis

# Ahora veremos como afecta la cuantizacion, tanto en pesos como en activaciones a la accuracy y loss

# In[ ]:


df = QuantizationEffect('AlexNet',test_set,wgt_dir,(227,227,3),8,test_batch_size)
#save_obj(df,'Data/Quantization/AlexNet/Colorectal Dataset/Quantization')


# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; b) Used Config

# En base a lo anterior parece prudente usar 11 bits de fraccion junto a 4 bits de parte entera tanto para pesos como activaciones, los resultados bajo esa configuracion se pueden obtener usando la funcion CheckAccurucyAndLoss()

# In[22]:


CheckAccuracyAndLoss('AlexNet', test_set, wgt_dir, act_frac_size = 11, act_int_size = 4, wgt_frac_size = 11, wgt_int_size = 4, 
                    input_shape = (227,227,3), output_shape = 8, batch_size = test_batch_size);


# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  c) Activation Stats

# Por otro lado, para la configuracion anterior, veremos el valor medio,maximo,minimo y el ratio de saturacion tanto de las activaciones procesadas dentro de la unidad matricial de multiplicacion como de las almacenadas en el buffer. Nota: el ultimo parametro indica el numero de iteraciones que se deben realizar hasta agotar el dataset, se obtiene como numero de imagenes del dataset dividido en el batch size.

# In[9]:


ActivationStats(AlexNet,test_set,11,4,24)


# # 3) Buffer Simulation

# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; a) Baseline

# Ahora para el Baseline simularemos el comportamiento de 1 buffer durante la inferencia de 3 imagenes (solo 3 como ejemplo), la red se crea ahora activando la cuantizacion pero no el envejecimiento. LI y AI son los definidos en el item 2) Stats

# In[23]:


train_batchSize = test_batchSize = 1
_,_,test_set = GetDatasets('colorectal_histology',(80,5,15),(227,227), 8, train_batchSize, test_batchSize)

#Obtain quantized network with the used configf
QAlexNet  = GetNeuralNetworkModel('AlexNet',(227,227,3),8, quantization = True, aging_active=False,
                                  word_size = 16, frac_size = 11)
loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
metrics = ['accuracy']
QAlexNet.compile(optimizer=optimizer, loss=loss, metrics=metrics)
QAlexNet.load_weights(wgt_dir).expect_partial()
#Quantize the weights of the network too, with used config
WeightQuantization(model = QAlexNet, frac_bits = 11, int_bits = 4)

#Lista de capas acorde a 2.b
LI = [0,3,9,11,17,19,25,31,37,40,45,50]
AI = [2,8,10,16,18,24,30,36,38,44,49,53]
Buffer,ciclos =  buffer_simulation(QAlexNet, test_set, integer_bits = 4, fractional_bits = 11, samples = 3, start_from = 0,
                                  bit_invertion = False, bit_shifting = False, CNN_gating = False,
                                  buffer_size = 2*290400, write_mode ='default', save_results = False,
                                  results_dir = 'Data/Stats/AlexNet/Colorectal Dataset/CNN-Gated/',
                                  layer_indexes = LI , activation_indixes = AI)


# El resultado es un diccionario como el siguiente donde por ejemplo Data contiene el ultimo valor registrado para cada celda de memoria.

# In[ ]:


Buffer


# In[ ]:


ciclos


# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; b) CNN-Gated, buffer 2 MB

# Se puede hacer lo mismo para distintas configuraciones de buffer y estrategias usadas

# In[ ]:


#train_batchSize = test_batchSize = 1
#_,_,test_set = GetDatasets('colorectal_histology',(80,5,15),(227,227), 8, train_batchSize, test_batchSize)
#
##Obtain quantized network with the used configf
#QAlexNet  = GetNeuralNetworkModel('AlexNet',(227,227,3),8, quantization = True, aging_active=False,
#                                  word_size = 16, frac_size = 11)
#loss = tf.keras.losses.CategoricalCrossentropy()
#optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
#metrics = ['accuracy']
#QAlexNet.compile(optimizer=optimizer, loss=loss, metrics=metrics)
#QAlexNet.load_weights(wgtDir).expect_partial()
##Quantize the weights of the network too, with used config
#WeightQuantization(model = QAlexNet, frac_bits = 11, int_bits = 4)
#
##Lista de capas acorde a 2.b
#LI = [0,3,9,11,17,19,25,31,37,40,45,50]
#AI = [2,8,10,16,18,24,30,36,38,44,49,53]
#Buffer,ciclos =  buffer_simulation(QAlexNet, test_set, integer_bits = 4, fractional_bits = 11, samples = 150, start_from = 0,
#                                  bit_invertion = False, bit_shifting = False, CNN_gating = True,
#                                  buffer_size = 2*1024*1024, write_mode ='default', save_results = False,
#                                  results_dir = 'Data/Stats/AlexNet/Colorectal Dataset/CNN-Gated/',
#                                  layer_indexes = LI , activation_indixes = AI)


# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; c) CNN-Gated, buffer ajustado a capa mas grande

# In[ ]:


#train_batchSize = test_batchSize = 1
#_,_,test_set = GetDatasets('colorectal_histology',(80,5,15),(227,227), 8, train_batchSize, test_batchSize)
#
##Obtain quantized network with the used configf
#QAlexNet  = GetNeuralNetworkModel('AlexNet',(227,227,3),8, quantization = True, aging_active=False,
#                                  word_size = 16, frac_size = 11)
#loss = tf.keras.losses.CategoricalCrossentropy()
#optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
#metrics = ['accuracy']
#QAlexNet.compile(optimizer=optimizer, loss=loss, metrics=metrics)
#QAlexNet.load_weights(wgtDir).expect_partial()
##Quantize the weights of the network too, with used config
#WeightQuantization(model = QAlexNet, frac_bits = 11, int_bits = 4)
#
#
#LI = [0,3,9,11,17,19,25,31,37,40,45,50]
#AI = [2,8,10,16,18,24,30,36,38,44,49,53]
#Buffer,ciclos =  buffer_simulation(QAlexNet, test_set, integer_bits = 4, fractional_bits = 11, samples = 150, start_from = 0,
#                                  bit_invertion = False, bit_shifting = False, CNN_gating = True,
#                                  buffer_size = 2*290400, write_mode ='default', save_results = False,
#                                  results_dir = 'Data/Stats/AlexNet/Colorectal Dataset/CNN-Gated/',
#                                  layer_indexes = LI , activation_indixes = AI)


# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ... d) para cargar los datos:

# tambien puedes cargar los datos desde la ruta que especificaste para guardar los resultados (parametro results_dir + save_results de buffer_simulation())

# In[ ]:


#Buffer  = load_obj('Data/Stats/AlexNet/Colorectal Dataset/CNN-Gated/Full Buffer/Buffer')
#ciclos  = load_obj('Data/Stats/AlexNet/Colorectal Dataset/CNN-Gated/Full Buffer/cycles')


# # 4) Error Injection

# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; a) Measuring effect of faults in activations

# por ultimo con este bloque vemos como se comporta el accuracy y loss frente a el envejecimiento de celdas.

# In[24]:


from copy import deepcopy
from Stats import CheckAccuracyAndLoss
from Simulation import save_obj, load_obj
from datetime import datetime
import itertools

trainBatchSize = testBatchSize = 16
_,_,test_dataset = GetDatasets('colorectal_histology',(80,5,15),(227,227), 8, trainBatchSize, testBatchSize)


# Porciones del buffer con fallos a probar
Accs     = {0.00001:[],0.00005:[],0.0001:[]}
Loss     = {0.00001:[],0.00005:[],0.0001:[]}

# Tamaño del buffer (en bits)
network_size   = 290400*16
# Numero de muestras de distintas configuraciones de fallos a testear por cada valor de Accs/Loss
num_of_samples = 10
for Enumber in Accs:
    n_bits_fails = np.ceil(Enumber*network_size).astype(int)    #numero de bits con fallos
    errors       = np.random.randint(0,2,n_bits_fails)          #tipo de fallos (0 o 1)
    # crear una representacion del buffer x indica celda inafectada, 1 celda con valor 1 permanente y 0 celda con valor 0 perm.
    buffer       = np.array(['x']*(network_size-n_bits_fails))  
    buffer       = np.concatenate([buffer,errors])
    for index in range(0,num_of_samples):
        np.random.shuffle(buffer)  # crear un orden aleatorio de los errores en el buffer
        # en las siguientes 4 lineas se obtienen las direcciones de los errores y los tipos de error
        address_with_errors = np.reshape(buffer,(-1,16))
        address_with_errors = ["".join(i) for i in address_with_errors]
        error_mask = [y for x,y in enumerate(address_with_errors) if y.count('x') < 16]
        locs       = [x for x,y in enumerate(address_with_errors) if y.count('x') < 16]
        del address_with_errors
        #ahora se obtiene el loss y acc para esta configuracion.
        loss,acc   = CheckAccuracyAndLoss('AlexNet', test_dataset, wgt_dir, output_shape=8, input_shape = (227,227,3),
                                            act_frac_size = 11, act_int_size = 4, wgt_frac_size = 11, wgt_int_size = 4,
                                            batch_size=testBatchSize, verbose = 0, aging_active = True, weights_faults = False,
                                            faulty_addresses = locs, masked_faults = error_mask)
        Accs[Enumber].append(acc)
        Loss[Enumber].append(loss)
    print(str(Enumber)+' completada: ', datetime.now().strftime("%H:%M:%S"))
    save_obj(Accs,'Data/Errors/AlexNet/Colorectal Dataset/Uniform distribution/Accs')
    save_obj(Loss,'Data/Errors/AlexNet/Colorectal Dataset/Uniform distribution/Loss')


# El resultado es un diccionario con accuracy y loss para cada muestra, por ejemplo, el caso de 0.0001 del buffer con fallos la accuracy ronda entre el 71 y 76% en los 10 casos aleatorios probados.

# In[ ]:


Accs


# In[ ]:


Loss


# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; a) Measuring effect of faults in weights

# Puedes hacer lo mismo para fallos en los pesos

# In[ ]:


from copy import deepcopy
from Stats import CheckAccuracyAndLoss
from Simulation import save_obj, load_obj
from datetime import datetime
import itertools


trainBatchSize = testBatchSize = 16
_,_,test_dataset = GetDatasets('colorectal_histology',(80,5,15),(227,227), 8, trainBatchSize, testBatchSize)


# Porciones del buffer con fallos a probar
Accs     = {0.00001:[],0.00005:[],0.0001:[]}
Loss     = {0.00001:[],0.00005:[],0.0001:[]}

# Tamaño del buffer (en bits)
network_size   = 885120*16
num_of_samples = 200
for Enumber in Accs:
    n_bits_fails = np.ceil(Enumber*290400).astype(int)
    errors       = np.random.randint(0,2,n_bits_fails)
    buffer       = np.array(['x']*(network_size-n_bits_fails))
    buffer       = np.concatenate([buffer,errors])
    for index in range(0,num_of_samples):
        np.random.shuffle(buffer)
        address_with_errors = np.reshape(buffer,(-1,16))
        address_with_errors = ["".join(i) for i in address_with_errors]
        error_mask = [y for x,y in enumerate(address_with_errors) if y.count('x') < 16]
        locs       = [x for x,y in enumerate(address_with_errors) if y.count('x') < 16]
        del address_with_errors
        loss,acc   = CheckAccuracyAndLoss('AlexNet', test_dataset, wgt_dir, output_shape=8, input_shape = (227,227,3),
                                            act_frac_size = 11, act_int_size = 4, wgt_frac_size = 11, wgt_int_size = 4,
                                            batch_size=testBatchSize, verbose = 0, aging_active = False, weights_faults = True,
                                            faulty_addresses = locs, masked_faults = error_mask)
        print(index,' completados: ', datetime.now().strftime("%H:%M:%S"))
        Accs[Enumber].append(acc)
        Loss[Enumber].append(loss)
    print(str(Enumber)+' completada: ', datetime.now().strftime("%H:%M:%S"))
    save_obj(Accs,'Data/Errors/AlexNet/Colorectal Dataset/Uniform distribution/weights Accs')
    save_obj(Loss,'Data/Errors/AlexNet/Colorectal Dataset/Uniform distribution/weights Loss')


# In[ ]:





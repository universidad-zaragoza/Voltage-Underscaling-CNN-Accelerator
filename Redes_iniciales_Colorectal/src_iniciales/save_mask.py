#!/usr/bin/env python
# coding: utf-8


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

# In[ ]:


import os
import pickle as pickle
import tensorflow as tf
import numpy as np
from Stats import WeightQuantization,IntroduceFaultsInWeights,GenerateFaultsList
from Nets import GetNeuralNetworkModel
from Training import GetDatasets
from Simulation import get_all_outputs
from Simulation import buffer_simulation, save_obj, load_obj
from funciones import compilNet, same_elements
import random


Cargar_errores = True
buffer_adress= (1024*1024)*1
#print(abuffer_size)

#print(load_file('Data/Fault Characterization/wgt/Accs_w_707_55'))

if Cargar_errores:
    locs  = load_obj('Data/Fault Characterization/error_mask y locs_buffer_act_vc-707/locs_0_54')
    #print(locs)
    print('mayor valor',max(locs))
    print('minimo valor',min(locs))
    
    error_mask = load_obj('Data/Fault Characterization/error_mask y locs_buffer_act_vc-707/error_mask_0_54')
    #locs  = load_obj('Data/Fault Characterization/error_mask y locs_buffer_pesos_vc-707/locs_0_54_buffer_pesos')
    #error_mask = load_obj('Data/Fault Characterization/error_mask y locs_buffer_pesos_vc-707/error_mask_0_54_buffer_pesos')
#print((error_mask))    
new_error_mask=(error_mask)*100
#print(new_error_mask)
rand_num_total=len(new_error_mask)-len(error_mask)
print(rand_num_total)
print('tamaño de error_mask', len(error_mask))
print('tamaño de new_error_mask', len(new_error_mask))    
random.seed(15)
print('locs antes',len(locs))
while len(locs)!= len(new_error_mask):
#for i in new_error_mask:
    new_locs= random.randint(0,buffer_adress )
    if new_locs not in (locs):
        locs.append(new_locs)
print('locs despues',len(locs)) 
#print(locs)
#print('mayor valor',max(locs))
#print('minimo valor',min(locs))
#print(len(locs))

 
dup = [x for i, x in enumerate(locs) if i != locs.index(x)]
print(dup) 
i=0
  
save_obj(locs,'Data/Fault Characterization/error_mask_x_100/error_mask_707/locs_0_54_buffer_act')
save_obj(new_error_mask,'Data/Fault Characterization/error_mask_x_100/error_mask_707/error_mask_0_54_buffer_act')
 
        


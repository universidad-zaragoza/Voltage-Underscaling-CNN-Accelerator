#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os
import pickle as pickle
import tensorflow as tf
import numpy as np
from Stats import WeightQuantization, IntroduceFaultsInWeights, GenerateFaultsList, CheckAccuracyAndLoss

from Training import GetPilotNetDataset
from Training import GetDatasets
from Simulation import get_all_outputs
from Simulation import buffer_simulation, save_obj, load_obj
from datetime import datetime
import collections
import pandas as pd
import os, sys
import pathlib

# In[14]:
# Acc_net: Desde una ruta definida con los ficgeros de error_mask y locs ya guardados con anterioridad para los voltajes analizados
# los recorre y calcula el acc por cada voltaje para cada red de las estudiadas, finalmente los huarda en un documento excel para un
# futuro análisis más detallado, si se desea hacer para otras máscaras de error basta con cambiar las direcciones donde se encuentren



# In[ ]:






# ficheros = [fichero.name for fichero in directorio.iterdir()]
# voltaj=('0.54','0.55','0.56','0.57','0.58','0.59','0.60')
voltaj = [54, 55, 56, 57, 58, 59, 60]
print(voltaj)
# voltaj=[56,58]
Voltajes = pd.DataFrame(voltaj)
print(Voltajes)
paso = 1
#
# #ficheros.sort()
#
vol = voltaj[0]


# # # AlexNet
#
# # In[24]:


cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'AlexNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir,'Weights')


Accs_B = []
Accs_I = []
Accs_E = []
Accs_F = []
Accs_F_P = []

trainBatchSize = testBatchSize = 1
_,_,test_dataset = GetDatasets('colorectal_histology',(80,5,15),(227,227), 8, trainBatchSize, testBatchSize)


# ruta_bin = 'Data/Fault Characterization/error_mask_0/vc_707'
# ruta_bin = 'Data/Fault Characterization/KC705_B/RowData/'
# directorio = pathlib.Path(ruta_bin)

# ficheros = [fichero.name for fichero in directorio.iterdir()]

# ficheros.sort()

vol = voltaj[0]
activation_aging = [False] * 11
for i in range(1):
    error_mask = load_obj(  'Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/error_mask_0' + str(vol))
    locs = load_obj(  'Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/locs_0' + str(vol))

    vol = vol + paso
    # print(i)
    # print('voltaje',vol)
    print('tamaño de locs', len(locs))

    loss, acc = CheckAccuracyAndLoss('AlexNet', test_dataset, wgt_dir, output_shape=8, input_shape=(227, 227, 3),
                                     act_frac_size=11, act_int_size=4, wgt_frac_size=11, wgt_int_size=4,
                                     batch_size=testBatchSize, verbose=0, aging_active=activation_aging,
                                     weights_faults=True,faulty_addresses=locs, masked_faults=error_mask)
    Accs_B.append(acc)
Base = pd.DataFrame(Accs_B)
print('Base')
print(Base)


vol = voltaj[0]
activation_aging = [False] * 11
for i in range(1):
    locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/ECCx4x1/ECCx/vc_707/locs_0'+ str(vol))
    error_mask = load_obj('Data/Fault Characterization/variante_mask_vc_707/ECCx4x1/ECCx/vc_707/error_mask_0'+ str(vol))

    vol = vol + paso
    # print(i)
    # print('voltaje',vol)
    print('tamaño de locs', len(locs))

    loss,acc   = CheckAccuracyAndLoss('AlexNet', test_dataset, wgt_dir, output_shape=8, input_shape = (227,227,3),
                                                act_frac_size = 11, act_int_size = 4, wgt_frac_size = 11, wgt_int_size = 4,
                                                batch_size=testBatchSize, verbose = 0, aging_active = activation_aging, weights_faults = True,
                                                faulty_addresses = locs, masked_faults = error_mask)

    Accs_I.append(acc)

iso_A = pd.DataFrame(Accs_I)
print('iso_A')
print(iso_A)



vol = voltaj[0]
activation_aging = [False] * 11
for i in range(1):
    locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/ECC/ECC/vc_707/locs_0'+ str(vol))
    error_mask = load_obj('Data/Fault Characterization/variante_mask_vc_707/ECC/ECC/vc_707/error_mask_0'+ str(vol))

    vol = vol + paso
    # print(i)
    # print('voltaje',vol)
    print('tamaño de locs', len(locs))

    loss, acc = CheckAccuracyAndLoss('AlexNet', test_dataset, wgt_dir, output_shape=8, input_shape=(227, 227, 3),
                                     act_frac_size=11, act_int_size=4, wgt_frac_size=11, wgt_int_size=4,
                                     batch_size=testBatchSize, verbose=0, aging_active=activation_aging,
                                     weights_faults=True, faulty_addresses=locs, masked_faults=error_mask)

    Accs_E.append(acc)

ECC = pd.DataFrame(Accs_E)
print('ECC')
print(ECC)



vol = voltaj[0]
activation_aging = [False] * 11
for i in range(1):
    locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/mask_volteada_x/mask_volteada/vc_707/locs_0'+ str(vol))
    error_mask = load_obj( 'Data/Fault Characterization/variante_mask_vc_707/mask_volteada_x/mask_volteada/vc_707/error_mask_0'+ str(vol))

    vol = vol + paso
    # print(i)
    # print('voltaje',vol)
    print('tamaño de locs', len(locs))

    loss, acc = CheckAccuracyAndLoss('AlexNet', test_dataset, wgt_dir, output_shape=8, input_shape=(227, 227, 3),
                                     act_frac_size=11, act_int_size=4, wgt_frac_size=11, wgt_int_size=4,
                                     batch_size=testBatchSize, verbose=0, aging_active=activation_aging,
                                     weights_faults=True, faulty_addresses=locs, masked_faults=error_mask)

    Accs_F.append(acc)

Flip = pd.DataFrame(Accs_F)
print('Flip')
print(Flip)



vol = voltaj[0]
activation_aging = [False] * 11
for i in range(1):
    locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/flip_patch/byte_x/vc_707/locs_0' + str(vol))
    error_mask = load_obj('Data/Fault Characterization/variante_mask_vc_707/flip_patch/byte_x/vc_707/error_mask_0' + str(vol))

    vol = vol + paso
    # print(i)
    # print('voltaje',vol)
    print('tamaño de locs', len(locs))

    loss, acc = CheckAccuracyAndLoss('AlexNet', test_dataset, wgt_dir, output_shape=8, input_shape=(227, 227, 3),
                                     act_frac_size=11, act_int_size=4, wgt_frac_size=11, wgt_int_size=4,
                                     batch_size=testBatchSize, verbose=0, aging_active=activation_aging,
                                     weights_faults=True,  faulty_addresses=locs, masked_faults=error_mask)
    Accs_F_P.append(acc)

F_P = pd.DataFrame(Accs_F_P)
print('F_P')
print(F_P)

print('Voltajes')
print(Voltajes)
buf_cero = pd.concat( [Voltajes, Base, iso_A, ECC, Flip, F_P,], axis=1, join='outer')
buf_cero.columns = ['Voltajes', 'Base', 'iso_A', 'ECC','Flip', 'F_P']
buf_cero.to_excel('Alexnet_weigts_acc_error_All_of_All.xlsx', sheet_name='fichero_707', index=False)


print('buf_cero', buf_cero)
print(str() + ' operación completada: ', datetime.now().strftime("%H:%M:%S"))





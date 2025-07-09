
import os
import pickle as pickle
import tensorflow as tf
import numpy as np
import collections
import pandas as pd
from Simulation import save_obj, load_obj
import pathlib

### Ciclo que recorra todas las mascras y locs y valla haciendo las estadisticas
##cuando termite convierto las listas de los datos que he ido guardando en dataframe y salvo en excel
# creo un script que ejecute primero el código de crear las máscaras y cdo termine ejecute este código y listo



cant_bit_les = []
cant_bit_more = []
cant_bit_conf = []
cant_elem =  []
no_mask=  []
no = 0
paso = 1

for f in range(2):
    no_mask.append(f)
    count_bit_les = 0
    coun_bit_more = 0
    count_bit_conf = 0
    error_mask_H_L_O = []
    locs_H_L_O = []
    vias = []

    locs= load_obj('MoRS/Modelo3_mas_fallos_col_16_sinx_delante/mask/locs_' + str(no))
    error_mask = load_obj('MoRS/Modelo3_mas_fallos_col_16_sinx_delante/mask/error_mask_' + str(no))
    for i, j in enumerate(error_mask):
        bit_more = j[0:8].count('0') + j[0:8].count('1')
        bit_les = j[8:16].count('0') + j[8:16].count('1')
        if bit_les > 0 and bit_more == 0:
            count_bit_les += 1

        elif bit_les == 0 and bit_more > 0:
            coun_bit_more += 1

        else:
            count_bit_conf += 1
            error_mask_H_L_O.append(j)
            locs_H_L_O.append(locs[i])
            vias.append(locs[i] % 256)

    unique, counts = np.unique(vias, return_counts=True)
    # print(dict(zip(unique, counts)))
    b = np.asarray(np.where(counts > 5))
    cant_elem.append(b.size)

    cant_bit_les.append(count_bit_les)
    #print('cant_bit_les', cant_bit_les)
    cant_bit_more.append(coun_bit_more)
    #print('cant_bit_more', cant_bit_more)
    cant_bit_conf.append(count_bit_conf)
    #print('cant_bit_conf', cant_bit_conf)

    Df_error_mask_H_L_O = pd.DataFrame(error_mask_H_L_O)
    #print('Df_error_mask_H_L_O',Df_error_mask_H_L_O)
    Df_locs_H_L_O = pd.DataFrame(locs_H_L_O)
    #print('Df_locs_H_L_O',Df_locs_H_L_O)
    Df_vias = pd.DataFrame(vias)
    #print('Df_vias',Df_vias)

    estad_H_L_O = pd.concat([Df_error_mask_H_L_O, Df_locs_H_L_O, Df_vias], axis=1, join='outer')
    estad_H_L_O.columns = ['error_mask_H_L_O', 'locs_H_L_O', 'vias']
    estad_H_L_O.to_excel('MoRS/Modelo3_mas_fallos_col_16_sinx_delante/estatics/Estad_H_L_O' + str(f) + '.xlsx', index=False)
    print('estad_H_L_O',estad_H_L_O)
    no = no + paso
    print('no',no)


df_mask = pd.DataFrame(no_mask)
print('df_mask',df_mask)
Df_cant_bit_les = pd.DataFrame(cant_bit_les)
# print('Df_cant_bit_les',Df_cant_bit_les)
Df_cant_bit_more = pd.DataFrame(cant_bit_more)
# print('Df_cant_bit_more',Df_cant_bit_more)
Df_cant_bit_conf = pd.DataFrame(cant_bit_conf)
# print('Df_cant_bit_conf',Df_cant_bit_conf)
Df_cant_elem = pd.DataFrame(cant_elem)
# print('Df_cant_elem',Df_cant_elem)

Tabla_resumen = pd.concat([df_mask,Df_cant_bit_les, Df_cant_bit_more, Df_cant_bit_conf, Df_cant_elem], axis=1, join='outer')
Tabla_resumen.columns = ['mask','LO', 'HO', 'L&HO', 'Vias']
print('Tabla_resumen',Tabla_resumen)
Tabla_resumen.to_excel('MoRS/Modelo3_mas_fallos_col_16_sinx_delante/estatics/Tabla_resumen_modelo.xlsx', index=False)
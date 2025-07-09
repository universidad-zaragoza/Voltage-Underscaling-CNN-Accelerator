#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import collections
import random
from random import randint
import os
import pickle


from datetime import datetime

def analize_file(obj_dir, buffer_size):
    with open(obj_dir , 'rb') as f:
        data = f.read()
    # Path(r'VC.bin').stat()
    # file_size = Path(r'VC.bin').stat().st_size
    index = 0
    buffer = []
    res = "{0:04b}".format(int(data, 16))
    res_list = list(res)
    t_error=res_list.count('0')
    #print('Cantidad de errores de todo el fihero:', t_error)
    vin_arr = []
    num_fallos = 0
    num_x = 0
    while index < buffer_size:

        for i, j in enumerate(res_list):

            if j == '1':
                vin_arr.append('x')
            else:
                vin_arr.append(0)
                num_fallos = num_fallos + 1
            index = index + 1
            if index == buffer_size:
                break
        buffer = np.asarray(vin_arr)
        #print('Cantidad de elementos por tipo :', collections.Counter(buffer))

        #print('numeros fallos', num_fallos)
    dist = collections.Counter(buffer)
    return buffer


def analize_file_uno(obj_dir, buffer_size):
    with open(obj_dir, 'rb') as f:
        data = f.read()
    # Path(r'VC.bin').stat()
    # file_size = Path(r'VC.bin').stat().st_size
    index = 0
    buffer = []
    res = "{0:04b}".format(int(data, 16))
    res_list = list(res)
    size_file=len(res_list)
    t_error = res_list.count('0')
    print('Cantidad de errores de todo el fihero:', t_error)
    print('Tamaño de todo el fihero:', size_file)

    vin_arr = []
    num_fallos = 0
    while index < buffer_size:

        for i, j in enumerate(res_list):

            if j == '1':
                vin_arr.append('x')

            else:

                vin_arr.append(1)

                num_fallos = num_fallos + 1
            index = index + 1
            if index == buffer_size:
                break
        print('tamaño de vin_arr:', len(vin_arr))
        buffer = np.asarray(vin_arr)
        print('Cantidad de elementos por tipo :', collections.Counter(buffer))
        print('tamaño de buffer:',len(buffer))

    print('numeros fallos', num_fallos)
    dist = collections.Counter(buffer)
    return buffer




def analize_file_uno_ceros(obj_dir, buffer_size):
    with open(obj_dir, 'rb') as f:
        data = f.read()
    # Path(r'VC.bin').stat()
    # file_size = Path(r'VC.bin').stat().st_size
    index = 0
    buffer = []
    res = "{0:04b}".format(int(data, 16))
    res_list = list(res)
    t_error = res_list.count('0')
    print('Cantidad de errores de todo el fihero:', t_error)

    vin_arr = []
    num_fallos = 0
    random.seed(15)

    while index < buffer_size:

        for i, j in enumerate(res_list):

            if j == '1':
                vin_arr.append('x')

            else:

                vin_arr.append(random.randint(0, 1))

                num_fallos = num_fallos + 1
            index = index + 1
            if index == buffer_size:
                break
        buffer = np.asarray(vin_arr)

        print('Cantidad de elementos por tipo :', collections.Counter(buffer))

    #print('numeros fallos', num_fallos)
    dist = collections.Counter(buffer)
    return buffer


def analize_file_full(obj_dir):
    with open(obj_dir, 'rb') as f:
        data = f.read()
    # Path(r'VC.bin').stat()
    # file_size = Path(r'VC.bin').stat().st_size
    index = 0
    buffer = []
    res = "{0:04b}".format(int(data, 16))
    res_list = list(res)
    size_file=len(res_list)
    t_error = res_list.count('0')
    print('Cantidad de errores de todo el fihero:', t_error)
    print('Tamaño de todo el fihero:', size_file)

    vin_arr = []
    num_fallos = 0
    while index < size_file:

        for i, j in enumerate(res_list):

            if j == '1':
                vin_arr.append('x')

            else:

                vin_arr.append(1)

                num_fallos = num_fallos + 1
            index = index + 1
            if index == size_file:
                break
        print('tamaño de vin_arr:', len(vin_arr))
        buffer = np.asarray(vin_arr)
        print('Cantidad de elementos por tipo :', collections.Counter(buffer))
        print('tamaño de buffer:',len(buffer))

    print('numeros fallos', num_fallos)
    dist = collections.Counter(buffer)
    return buffer,size_file

def buffer_vectores(buffer):
    buffer_size=len(buffer)
    address_with_errors = np.reshape(buffer, (-1, 16))
    address_with_errors = ["".join(i) for i in address_with_errors]
    error_mask = [y for x,y in enumerate(address_with_errors) if y.count('x') < 16]
    locs       = [x for x,y in enumerate(address_with_errors) if y.count('x') < 16]
    del address_with_errors
    #address_with_errors = ["".join(i) for i in address_with_errors[0:2000]]
    #error_mask = [y for x, y in enumerate(address_with_errors) if y.count('x') < 16]
    #locs = [x for x, y in enumerate(address_with_errors) if y.count('x') < 16]

    #del address_with_errors
    return [error_mask, locs]


def save_file(obj, obj_dir):
    with open(obj_dir + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_file(obj_dir):
    with open(obj_dir + '.pkl', 'rb') as f:
        return pickle.load(f)



## esta función voltea los vectores que cumplen con la condición que los bits menos sinificativo siguen
## el patrón 'xxxx' y los más significativos contienen errores ya sea por '1' o por '0'

def FlipMask(error_mask,locs):
    error_mask_flip = []
    locs_flip = []
    count_flip = 0
    marca = '*'
    print('máscara original', error_mask[21:30])
    len_error_mask = len(error_mask)
    print('Tamaño de máscara ', len_error_mask)
    for i, j in enumerate(error_mask):
        # print(j[0:4])

        if '0' in j[0:4] or '1' in j[0:4] and j.endswith('xxxx'):
            error_volteado = (j[::-1])
            error_mask_flip.append(error_volteado)
            count_flip = count_flip + 1
            locs_flip.append(locs[i])
        else:
            error_mask_flip.append(j)
    wth_flip = len_error_mask - count_flip
    print('cantidad volteada', count_flip)
    print('cantidad sin voltear', wth_flip)

    print('Máscara volteada', error_mask_flip[21:30])
    print('tamaño de Máscara volteada ', len(error_mask_flip))
    return error_mask_flip, locs_flip

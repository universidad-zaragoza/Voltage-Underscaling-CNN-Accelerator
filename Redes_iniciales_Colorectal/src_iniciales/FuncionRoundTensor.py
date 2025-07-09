


from tensorflow.keras.models import Sequential
from tensorflow.python.keras import backend as K
import tensorflow as tf
import collections
import numpy as np
from openpyxl import load_workbook
from openpyxl import Workbook
from pandas import ExcelWriter
import pandas.io.formats.excel
from funciones import TensorUpdateCiclo






# In[7]:





# ## Declaración de variables y preparación de los datos que entran al ciclo

# In[ ]:


#values_to_run:
#mask
#v_m: Los valores dentro del tensor con error afectado por los errores
#v_b: Los valores dentro del tensor original no afectado por los errores
#mask_0: Lo creo como una constante que luego va sumando los cambios que hago al final no es necesaria pero la dejé por si acaso
#valores_afectados: tensor de (T and F),donde en T están los valores afectados por los errores inyectados # esta matriz es la que controla si ya no hay cambios pendientes porque todos los valores han sido analizados y
#detiene el ciclo para no iterar innecesariamente

#valores_afec_static_0_Error: los errores que quedan lugar a cambios dentro del tensor con error
#valores_afec_static_1_Error:
#vb_and_mask
#vm_and_mask
#tensor_act
#values_to_run


# In[ ]:



# In[26]:
def ErrorAcero(values_to_change, mask, val_afec_0_Error_to_change, val_afec_1_Error_to_change):
    print('dentro de error a 0')

    # tensor_final=tf.where(tf.math.not_equal(vb_and_mask,vm_and_mask) )
    n_mask = mask - 1
    print('mask', mask)
    print('n_mask', n_mask)
    tensor_round = tf.bitwise.bitwise_or(values_to_change, n_mask)
    print('n_v_b_m', tensor_round)
    # tensor=tf.bitwise.bitwise_and(n_v_b_m,val_afec_0_Error_to_change)
    # print('n_v_b_1', tensor)
    # tensor_round=tf.bitwise.bitwise_or(tensor,val_afec_1_Error_to_change)
    # print( 'tensor_round',tensor_round)
    # tensor_round  = tf.where(tf.greater_equal(tensor_round,shift), shift-tensor_round , tensor_round )

    return tensor_round


# En este caso sele resta 1 a la máscara pero debe hacerse un not_lógico a la misma para para cuando se haga el btw_and
# colocar el retso de los bit luego de la posición con error a 0, lo demás es como lo anterior.
def ErrorAuno(values_to_change, mask, val_afec_0_Error_to_change, val_afec_1_Error_to_change):
    print('dentro de error a 1`')

    # tensor_final=tf.where(tf.math.not_equal(vb_and_mask,vm_and_mask) )
    n_mask = mask - 1
    print('mask', mask)
    print('n_mask', n_mask)
    not_mask = np.invert(np.array([n_mask], dtype=np.uint16))
    print('not_mask', not_mask)
    tensor_round = tf.bitwise.bitwise_and(values_to_change, not_mask)
    print('n_v_b_m', tensor_round)
    # tensor=tf.bitwise.bitwise_and(n_v_b_m,val_afec_0_Error_to_change)
    # print('n_v_b_1', tensor)
    # tensor_round=tf.bitwise.bitwise_or(tensor,val_afec_1_Error_to_change)
    # tensor_round  = tf.where(tf.greater_equal(tensor_round,shift), shift-tensor_round , tensor_round )
    return tensor_round




def FuncionRound(tensor,faults):
    tensor = tf.bitwise.bitwise_and(tensor, faults[:, 0])
    print('faults[:,0]', faults[:, 0])
    tensor_with_error = tf.bitwise.bitwise_or(tensor, faults[:, 1])
    tensor_with_error
    values_to_run=tf.Variable(0)
    mask=32768
    mask=tf.convert_to_tensor(mask)
    valores_afectados=tf.math.not_equal(tensor,tensor_with_error)
    print('valores_afectados',valores_afectados)
    v_m=tf.gather_nd(tensor_with_error,tf.where(valores_afectados==True))
    print('valores afectados por los errores' ,v_m)
    v_b=tf.gather_nd(tensor,tf.where(valores_afectados==True))
    print('valores que se deberian escribir' ,v_b)
    mask_0=tf.constant(0)
    print('mask_0',mask_0)
    i=tf.constant(0)
    out=tf.constant(16)

    valores_afec_static_0_Error=tf.gather_nd(faults[:,0], tf.where(tf.math.not_equal(tensor,tensor_with_error)))
    valores_afec_static_1_Error=tf.gather_nd(faults[:,1], tf.where(tf.math.not_equal(tensor,tensor_with_error)))
    vb_and_mask=tf.bitwise.bitwise_and(tf.gather_nd(tensor,tf.where(valores_afectados==True)),mask)
    vm_and_mask=tf.bitwise.bitwise_and(tf.gather_nd(tensor_with_error,tf.where(valores_afectados==True)),mask)
    tensor_act=tensor
    a=tf.math.not_equal(vb_and_mask,vm_and_mask)
    b = tf.math.greater(v_m, v_b)  # El error es a 1 y aplico la variante con_error_a_1
    print('b Vm>Vb', b)
    c = tf.where(tf.logical_and(a, b) == True)

    print('c', c)
    print('tamaño de c', tf.size(c))
    print(' c Indices de los valores a transformar a 0', c)
    index = tf.experimental.numpy.array(c)
    #index = c.numpy()
    #index = tf.constant(index)
    print('cnumpy', index)
    # print('valores_afectados',valores_afectados)
    val = tf.gather_nd(tf.where(valores_afectados), index)  ## tomo los indices del tensor original qu ese cambiaran

    error_a_0 = tf.math.greater(v_b, v_m)
    index_values_error_a_0 = tf.where(tf.logical_and(a, error_a_0) == True)
    print('index_values_error_a_0', index_values_error_a_0)
    #index_0 = index_values_error_a_0.numpy()
    index_0 = tf.experimental.numpy.array(index_values_error_a_0)
    print('index_0', index_0)
    print('valores_afectados', valores_afectados)
    val_0 = tf.gather_nd(tf.where(valores_afectados), index_0)  ## tomo los indices del tensor original qu ese cambiaran
    print('val_0', val_0)
    tensor_act = tf.tensor_scatter_nd_update(tensor_act, val, tf.convert_to_tensor([0] * tf.size(val)))
    tensor_act = tf.tensor_scatter_nd_update(tensor_act, val_0, tf.convert_to_tensor([0] * tf.size(val_0)))
    # join = tf.concat([val, val_0], 0)
    # join = tf.cast(join, dtype=tf.int32)
    # print('join',join)
    #join = tf.experimental.numpy.array(join)
    #join

    # def f1():
    #     return TensorUpdatePosicionInicial(tensor_act, join, valores_afectados, mask_0, tensor_with_error, faults,index)
    #
    # def f2():
    #     return Ciclo()
    # def output(val, tensor_act):
    #     for i in range(1):
    #         if tf.size(val) > 0:
    #             print('en el if val')
    #             tensor_act = tf.tensor_scatter_nd_update(tensor_act, tf.gather_nd(tf.where(valores_afectados), index),
    #                                                      tf.convert_to_tensor([0] * tf.size(val)))
    #             tensor_act = tf.tensor_scatter_nd_update(tensor_act, tf.gather_nd(tf.where(valores_afectados),
    #                                                                               index_values_error_a_0),
    #                                                      tf.convert_to_tensor([0] * tf.size(val)))
    #tensor_act, valores_afectados, mask_0, v_b, v_m, valores_afec_static_0_Error, valores_afec_static_1_Error = tf.cond(tf.size(join) > 0, f1, f2)
    #print(tensor_act)
    #mask = tf.bitwise.right_shift(mask, 1)

#     if tf.reduce_sum(mask_1_ini)> 0:
#         tensor_act, valores_afectados, mask_0, v_b, v_m = TensorUpdatePosicionInicial(
#             tensor_act, val_0, valores_afectados, mask_0, tensor_with_error, faults)
#
# ## entra a este if cuando hay error a 1 en la posición más significativa
#     if tf.reduce_sum(mask_0_ini)> 0:
#         tensor_act, valores_afectados, mask_0, v_b, v_m = TensorUpdatePosicionInicial(
#             tensor_act, val, valores_afectados, mask_0, tensor_with_error, faults)

    # if tf.size(val) > 0:
    #     tensor_act, valores_afectados, mask_0, v_b, v_m, valores_afec_static_0_Error, valores_afec_static_1_Error = TensorUpdatePosicionInicial(
    #         tensor_act, val, valores_afectados, mask_0, tensor_with_error, faults)
    #
    # if tf.size(val_0) > 0:
    #     tensor_act, valores_afectados, mask_0, v_b, v_m, valores_afec_static_0_Error, valores_afec_static_1_Error = TensorUpdatePosicionInicial(
    #         tensor_act, val_0, valores_afectados, mask_0, tensor_with_error, faults)

    #
    print('tensor_act',tensor_act)
    print('valores_afectados',valores_afectados)
    print('tensor_with_error',tensor_with_error)
    print('v_b',v_b)
    print('v_m',v_m)




# ## modificarlo porque en el caso donde hay 0 en la posición 1 no lo estaba contemplando

# In[ ]:


#values_to_run_with_error
#valores_afec_static_0_Error_to_run
#valores_afec_static_0_Error_to_run
#values_to_change


# In[31]:

    @tf.function
    def while_body(i, out, mask, mask_0, tensor, tensor_act, v_m, v_b, tensor_with_error, values_to_run, valores_afectados,  valores_afec_static_0_Error, valores_afec_static_1_Error,get_shape):
        print('dentro del body')


        #     print('i',i)
        #     print('out que llega al body', out)
        #      print('tensor_act',tensor_act)
        # print('tensor',tensor)
        # print('mask',mask)
        # print('mask_0',mask_0)
        # print('v_m',v_m)
        # print('v_b',v_b)
        # print('tensor_with_error',tensor_with_error)
        # hay_cambios=tf.math.not_equal(tensor,tensor_act)
        # print('hay_cambios',hay_cambios)
        # stop_condition=tf.experimental.numpy.any(valores_afectados)
        # if stop_condition==False:

        ## Este if lo cree por si hay cambios antes de llegar al ciclo por haber errores e la posición 15 pero quizás ya o sea
        # necesraio analizar luego de experimentar  si mask_0 llega con valores a 1 es porque se hicieron cambios y se deben
        # analizar los btw con los valores resttantes sino todo se mantiene con los valores originales de entrada.
        #@tf.function
        #def Condicional(mask_0,v_b,v_m,mask,valores_afectados,valores_afec_static_0_Error,valores_afec_static_1_Error,tensor_act):
        if tf.reduce_sum(mask_0) > 0:
                # print('entre en condicion mask_0>0')
                # print('valores donde debo correr el mask')

            vb_and_mask = tf.bitwise.bitwise_and(v_b, mask)
            print('vb_and_mask', vb_and_mask)
            vm_and_mask = tf.bitwise.bitwise_and(v_m, mask)
            print('vm_and_mask', vm_and_mask)
        else:
            print('entre en el else')
            vb_and_mask = tf.bitwise.bitwise_and(tf.gather_nd(tensor, tf.where(mask_0 == True)), mask)
            print('vb_and_mask', vb_and_mask)
            vm_and_mask = tf.bitwise.bitwise_and(tf.gather_nd(tensor_with_error, tf.where(valores_afectados == True)),
                                                 mask)
            print('vm_and_mask', vm_and_mask)
            print('v_b', v_b)
            print('v_m', v_m)

        a = tf.math.not_equal(vb_and_mask, vm_and_mask)
        values_to_run = tf.gather_nd(v_b, tf.where(a == False))
        values_to_run_with_error = tf.gather_nd(v_m, tf.where(a == False))
        valores_afec_static_0_Error_to_run = tf.gather_nd(valores_afec_static_0_Error, tf.where(a == False))
        valores_afec_static_1_Error_to_run = tf.gather_nd(valores_afec_static_1_Error, tf.where(a == False))
        b = tf.math.greater(v_m, v_b)  # El error es a 1 y aplico la variante con_error_a_1

        print('b Vm>Vb', b)
        c = tf.where(tf.logical_and(a, b) == True)
               # c=tf.where(tf.equal(a,b))
        error_a_0 = tf.math.greater(v_b, v_m)
        print('error_a_0', error_a_0)
        index_values_error_a_0 = tf.where(tf.logical_and(a, error_a_0) == True)
        # index_values_error_a_0=tf.where(error_a_0==True)
        print('index_values_error_a_0', index_values_error_a_0)
        print('vb', v_b)

        if tf.size(c) > 0:
            error = 1
            tensor_act, valores_afectados, mask_0, val_afec_0_Error_to_change, val_afec_1_Error_to_change = TensorUpdateCiclo(
                error, v_b, c, valores_afec_static_0_Error,  valores_afec_static_1_Error, mask, tensor_act,
                valores_afectados, mask_0)


        ##no es el que esta en mask en esse momento, retorno new mask, new_mask_0, tensor_act
        if tf.size(index_values_error_a_0) > 0:
            error = 0
            print('dentro detf.size(error_a_0)>0 ')
            tensor_act, valores_afectados, mask_0, val_afec_0_Error_to_change, val_afec_1_Error_to_change = TensorUpdateCiclo(
            error, v_b, index_values_error_a_0, valores_afec_static_0_Error, valores_afec_static_1_Error, mask,
            tensor_act, valores_afectados, mask_0)

            print('tensor tensor_act', tensor_act)
            print('valores_afectados update', valores_afectados)
                # mask_0=tf.add(mask_0,tf.size(val_0))
            print('mask_0 desdes update', mask_0)

        if tf.size(values_to_run) > 0:
                # mask=mask>>1
            mask = tf.bitwise.right_shift(mask, 1)
            print('dentro del if values to run', values_to_run)
            print('mask', mask)
            v_b = values_to_run
            v_m = values_to_run_with_error
            valores_afec_static_0_Error = valores_afec_static_0_Error_to_run
            valores_afec_static_1_Error = valores_afec_static_1_Error_to_run
            print('v_b', v_b)
            print('v_m', v_m)


        #if tf.experimental.numpy.any(valores_afectados) == False:

            #print('dentro de condicion de parada')


            #print('valores_afectados', valores_afectados)
            #print('tensor_act', tensor_act)
            #i = 16


        print('i', i)
        print('0ut',out)

        return (i + 1), out, mask, mask_0, tensor, tensor_act, v_m, v_b, tensor_with_error, values_to_run, valores_afectados, valores_afec_static_0_Error, valores_afec_static_1_Error,i.get_shape()  ## retorno los valores que aun no se han modificado por ninguna d elas variantes puesto que el bit con error
    
    i, out, mask, mask_0, tensor, tensor_act, v_m, v_b, tensor_with_error, values_to_run, valores_afectados, valores_afec_static_0_Error, valores_afec_static_1_Error,shape_invariants = tf.while_loop(
    lambda i, *_: tf.less(i, out),  # condición -> revisar cada bit
    while_body,  # aqui
            # o pasarlos a la misma función de while_body y no modificarlos)
    (i, out, mask, mask_0, tensor, tensor_act, v_m, v_b, tensor_with_error, values_to_run, valores_afectados,  valores_afec_static_0_Error, valores_afec_static_1_Error,i.get_shape())  # valores iniciales
    )

#return tensor_act
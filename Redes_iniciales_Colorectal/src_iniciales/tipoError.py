import tensorflow as tf
import numpy as np


@tf.function
def ErrorAcero(values_to_change, mask):
    print('dentro de error a 0')

    # tensor_final=tf.where(tf.math.not_equal(vb_and_mask,vm_and_mask) )
    n_mask = mask - 1
    print('mask', mask)
    print('n_mask', n_mask)
    tensor_round = tf.bitwise.bitwise_or(values_to_change, n_mask)
    print('n_v_b_m', tensor_round)


    return tensor_round


# En este caso sele resta 1 a la máscara pero debe hacerse un not_lógico a la misma para para cuando se haga el btw_and
# colocar el retso de los bit luego de la posición con error a 0, lo demás es como lo anterior.
@tf.function
def ErrorAuno(values_to_change, mask):
    print('dentro de error a 1`')

    # tensor_final=tf.where(tf.math.not_equal(vb_and_mask,vm_and_mask) )
    n_mask = mask - 1
    print('mask', mask)
    print('n_mask', n_mask)
    #not_mask = np.invert(np.array([n_mask], dtype=np.uint16))
    #np_mask= tf.experimental.numpy.array([n_mask]
    not_mask=tf.bitwise.invert(tf.experimental.numpy.array([n_mask], dtype=np.uint16))
    print('not_mask', not_mask)
    #tf.cast(not_mask, dtype=tf.int32)
    tensor_round = tf.bitwise.bitwise_and(values_to_change,  tf.cast(not_mask, dtype=tf.int32))
    print('n_v_b_m', tensor_round)

    return tensor_round

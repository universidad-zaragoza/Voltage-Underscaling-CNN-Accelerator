from tensorflow.keras.layers import (Activation, AveragePooling2D, BatchNormalization, Cropping2D,
                                     Concatenate, Conv1D, Conv2D, Dense, DepthwiseConv2D, Dropout,
                                     Embedding, Flatten, GlobalAveragePooling2D, Lambda, MaxPool2D,
                                     MaxPooling1D, ReLU, Reshape, ZeroPadding2D,Add,MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.python.keras import backend as K
import tensorflow as tf
import collections
import pandas as pd
import numpy as np
#from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D,\
     #Flatten, BatchNormalization, AveragePooling2D, Dense, Activation, Add
from tensorflow.keras.models import Model
from tensorflow.keras import activations
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras import backend as K
import tensorflow as tf
###################################################################################################
##FUNCTION NAME: GetNeuralNetworkModel
##DESCRIPTION:   Crea la red neuronal segun las especificaciones requeridas
##OUTPUTS:       tf.keras.Model: Red Neuronal especificada
############ARGUMENTS##############################################################################
####architecture:     Modelo de la Red, uno de los siguientes string: 'AlexNet','VGG16','PilotNet',
####                  'MobileNet','ZFNet','SqueezeNet','SentimentalNet','DenseNet'
####input_shape:      Dimensiones de entrada de la red, ejemplo: (228,228)
####                  el resto por defecto es para testing.
####output_shape:     Dimensiones de salida de la red, ejemplo: 10
####faulty_addresses: Lista con las direcciones de memoria que contienen errores,
####                  ejemplo: [1,1024,203405]
####masked_faults:    Lista de las fallas para las direcciones especificadas en faulty_addresses,
####                  ejemplo: ['xx1xxxxxxxxxxxxx','1xxxxxxxxxxxxxx0', '0000000000000000'],
####                  '1'/'0' = celdas con valor 1 o 0 permanente, 'x' = celda sin fallos.
####quantization:     Contruir el modelo cuantizado o no de la red, si es Falso word_size
####                  y frac_size no son usados.
####aging_active:            Contruir el modelo con fallos o no de la red, si es Falso faulty_addresses
####                  y masked_faults no son usados
####word_size:        Tamaño en bits de una activacion
####frac_size:        Numero de bits para la parte fraccionario de una activacion.
####batch_size:       Tamaño de batch para inferencia.
###################################################################################################

# def GetNeuralNetworkModel(architecture: object, input_shape: object, output_shape: object, faulty_addresses: object = [], masked_faults: object = [],
#                           quantization: object = True, aging_active: object = False, word_size: object = None, frac_size: object = None, batch_size: object = 1) -> object:
def GetNeuralNetworkModel(architecture, input_shape, output_shape, faulty_addresses=[],  masked_faults=[], quantization=True, aging_active=False,
							  word_size=None, frac_size=None, batch_size=1):
	print('dentro de getneuronalmodel')
	# Layer to quantize the values to the magnitude sign format when needed
	def preprocess(image):  # preprocess image
		return tf.image.resize(image, (200, 66))
    ###############################################################################################
    ##FUNCTION NAME: Quantization
    ##DESCRIPTION:   Cuantiza un tensor usando un numero fijo de bits de signo, magnitud y fraccion
    ##OUTPUTS:       Tensor cuantizado
    ############ARGUMENTS##########################################################################
    ####tensor:    Tensorflow tensor
    ####active:    Habilita la cuantizacion
    ###############################################################################################
	def Quantization(tensor, active = True):
		if active:
			factor = 2.0**frac_size
			max_value = ((1 << (word_size-1)) - 1)/factor
			min_value = -max_value
			tensor = tf.round(tensor*factor) / factor
			tensor = tf.math.minimum(tensor,max_value)   # Upper Saturation
			tensor = tf.math.maximum(tensor,min_value)   # Lower Saturation
		return tensor 
	###############################################################################################
    ##FUNCTION NAME: Aging
    ##DESCRIPTION:   Aplica envejecimiento a un tensor segun su mapeo en memoria
    ##OUTPUTS:       Tensor con valores alterados debido a los fallos
    ############ARGUMENTS##########################################################################
    ####tensor:     Tensorflow tensor
    ####index_list: tensor con indices de las activaciones afectadas por los fallos
    ####mod_list:   tensor con fallos a aplicar usando bitwise & |.
    ####active:     Habilita el envejecimiento
    ###############################################################################################
	def UnderVolting(tensor, index_list, mod_list, active):
		#print('UnderVolting')
		affectedValues_array = []
		newValues_array = []
		def ApplyFault(tensor,faults):
			#print('dentro de ApplyFault')
			Ogdtype = tensor.dtype
			shift   = 2**(word_size-1)
			factor  = 2**frac_size
			tensor  = tf.cast(tensor*factor,dtype=tf.int32)
			tensor  = tf.where(tf.less(tensor, 0), -tensor + shift , tensor )
			tensor  = tf.bitwise.bitwise_and(tensor,faults[:,0])
			tensor  = tf.bitwise.bitwise_or(tensor,faults[:,1])
			tensor  = tf.where(tf.greater_equal(tensor,shift), shift-tensor , tensor )
			tensor  = tf.cast(tensor/factor,dtype = Ogdtype)
			return tensor		
		if active:
			#print('Experimento Base')
			affectedValues = tf.gather_nd(tensor,index_list)
			affectedValues_array.append(affectedValues)
			newValues = ApplyFault(affectedValues,mod_list)
			newValues_array.append(newValues)
			diff_abs = tf.math.reduce_sum(tf.math.abs(tf.math.subtract(affectedValues_array, newValues_array)))
			#print('diferencia absoluta', diff_abs)
			tensor = tf.tensor_scatter_nd_update(tensor, index_list, newValues)

			
		return tensor
	###############################################################################################
    ##FUNCTION NAME: GenerateAddressList
    ##DESCRIPTION:   Mapea la lista de direcciones de memoria con fallos y la lista de tipos de 
    ##               fallos enmascarados a una tensor de indices de activaciones con fallos y 
    ##               un tensor de fallos.
    ##OUTPUTS:       tensor de indices, tensor de fallos, numero de activaciones afectadas
    ############ARGUMENTS##########################################################################
    ####shape:       Dimensiones de entrada de la capa
    ###############################################################################################
	total_error_capa = []
	index_list_capa = []
	def GenerateAddressList(shape):
		# Decodes the mask of faults to the specific error due to 0 and 1 static value
		def DecodeMask(mask):
			#print('dentro de DecodeMask ')
			static_1_error  = int("".join(mask.replace('x','0')),2)
			static_0_Error  = int("".join(mask.replace('x','1')),2)
			return [static_0_Error,static_1_error]
		index_list   = []
		mod_list      = []
		error_zero = []
		error_uno = []
		if len(shape) == 1:
			for index in range(batch_size):
				for address,mask in zip(faulty_addresses,masked_faults):
					if address < shape[0]-1:
						index_list.append([index,address])
						mod_list.append(DecodeMask(mask))
						error_zero.append(collections.Counter(mask)['0'])
						error_uno.append(collections.Counter(mask)['1'])
		elif len(shape) == 2:
			for index in range(batch_size):
				for address,mask in zip(faulty_addresses,masked_faults):
					if address < shape[0]*shape[1] - 1:
						Ch1  = address//shape[0]
						Ch2  = (address - Ch1*shape[0])//shape[1]
						index_list.append([index,Ch1,Ch2])
						mod_list.append(DecodeMask(mask))
						error_zero.append(collections.Counter(mask)['0'])
						error_uno.append(collections.Counter(mask)['1'])
		else:
			for index in range(batch_size):
				for address,mask in zip(faulty_addresses,masked_faults):
					if address < shape[0]*shape[1]*shape[2] - 1:
						actMap = address//(shape[0]*shape[1])
						row    = (address - actMap*shape[0]*shape[1])//shape[1]
						col    = address - actMap*shape[0]*shape[1] - row*shape[1]
						index_list.append([index,row,col,actMap])
						mod_list.append(DecodeMask(mask))
						error_zero.append(collections.Counter(mask)['0'])
						error_uno.append(collections.Counter(mask)['1'])
		faults_count = len(index_list)
		# index_list_capa.append(faults_count)
		# error_0 = sum(error_zero)
		# error_1 = sum(error_uno)
		# total_error = error_0 + error_1
		# total_error_capa.append(error_0 + error_1)
		# print('suma de todos los errores', sum(total_error_capa))
		# df_index_list_capa = pd.DataFrame(index_list_capa)
		# df_total_error_capa = pd.DataFrame(total_error_capa)
		# df_error = pd.concat([df_index_list_capa, df_total_error_capa], axis=1, join='outer')
		# df_error.columns = ['fail_act', 'a_error']
		# print('dentro de net guardando excel')
		# df_error.to_excel('resul_F_P'+ str(architecture)+'.xlsx' ,sheet_name=architecture, index=False)
		#with open('AlexNet/ratio_element_diff_pruebas.xlsx') as f:
		#writer = pd.ExcelWriter('AlexNet/actv_error' + str(architecture) + '.xlsx', engine='xlsxwriter')
		#df_error.to_excel(writer, sheet_name=architecture, index=False)

		#writer.save()
		return tf.convert_to_tensor(index_list),tf.convert_to_tensor(mod_list), faults_count
	###############################################################################################
    ##FUNCTION NAME: AddCustomLayers
    ##DESCRIPTION:   Agrega a una red neuronal capas de cuantizacion y/o envejecimiento
    ##OUTPUTS:       Capa de red neuronal
    ############ARGUMENTS##########################################################################
    ####input_layer:          Capa de red neuronal a la cual agregarle cuantizacion/aging_active
    ####include_aging:        True  si se desea incorporar una capa de aging_active
    ####include_quantization: True  si se desea incorporar una capa de cuantizacion
    ####aging_active:         False si se desea que la capa de aging_active no este activa
    ###############################################################################################
	def AddCustomLayers(input_layer, include_aging, include_quantization=True, aging_active = []):

		x = input_layer
		if include_quantization:
			quantization_arguments = {'active':quantization}
			x = Lambda(Quantization, arguments = quantization_arguments)(input_layer)
		if include_aging:
			dims = x.shape.ndims if x.__class__.__name__ == 'KerasTensor' else x.output_shape.ndim
			index_list, mod_list, faults_count = GenerateAddressList(shape=x.shape[1:])
			aging_arguments = {'index_list': tf.identity(index_list),
							   'mod_list': tf.identity(mod_list),
							   'active': tf.identity(faults_count and aging_active)}
			x = Lambda(UnderVolting, arguments=aging_arguments)(x)
		return x

	# if   aging_active == True:  aging_active = [True]*300
	# elif aging_active == False: aging_active = [False]*300
	#AlexNet
	if architecture=='AlexNet':
		input_layer = tf.keras.Input(input_shape)
		x = AddCustomLayers(input_layer,include_aging=True,aging_active = aging_active[0])
		x = Conv2D(filters=96, kernel_size=(11,11), strides=(4,4),name='Conv1')(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = BatchNormalization()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[1])
		x = MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[2],include_quantization=False)
		x = Conv2D(filters=256, kernel_size=(5,5), strides=(1,1),padding="same",name='Conv2')(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = BatchNormalization()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[3])
		x = MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[4],include_quantization=False)
		x = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="same",name='Conv3')(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = BatchNormalization()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[5])
		x = Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), padding="same",name='Conv4')(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = BatchNormalization()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[6])
		x = Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), padding="same",name='Conv5')(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = BatchNormalization()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[7])
		x = MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[8],include_quantization=False)
		x = Flatten()(x)
		x = Dense(4096)(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = Dropout(0.5)(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[9],include_quantization=False)
		x = Dense(4096)(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = Dropout(0.5)(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[10],include_quantization=False)
		x = Dense(output_shape)(x)
		x = AddCustomLayers(x,include_aging=False)
		x = tf.keras.activations.softmax(x)
		x = AddCustomLayers(x,include_aging=False)
		Net = tf.keras.Model(inputs=input_layer, outputs=x)
		return Net

	elif architecture=='VGG16':
		input_layer = tf.keras.Input(input_shape)
		x = AddCustomLayers(input_layer,include_aging=True,aging_active = aging_active[0])
		x = Conv2D(filters=64,kernel_size=(3,3),padding="same")(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[1],include_quantization=False)
		x = Conv2D(filters=64,kernel_size=(3,3),padding="same")(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[2],include_quantization=False)
		x = MaxPool2D(pool_size=(2,2),strides=(2,2))(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[3],include_quantization=False)
		x = Conv2D(filters=128,kernel_size=(3,3),padding="same")(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[4],include_quantization=False)
		x = Conv2D(filters=128,kernel_size=(3,3),padding="same")(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[5],include_quantization=False)
		x = MaxPool2D(pool_size=(2,2),strides=(2,2))(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[6],include_quantization=False)
		x = Conv2D(filters=256,kernel_size=(3,3),padding="same")(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[7],include_quantization=False)
		x = Conv2D(filters=256,kernel_size=(3,3),padding="same")(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[8],include_quantization=False)
		x = Conv2D(filters=256,kernel_size=(3,3),padding="same")(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[9],include_quantization=False)
		x = MaxPool2D(pool_size=(2,2),strides=(2,2))(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[10],include_quantization=False)
		x = Conv2D(filters=512,kernel_size=(3,3),padding="same")(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[11],include_quantization=False)
		x = Conv2D(filters=512,kernel_size=(3,3),padding="same")(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[12],include_quantization=False)
		x = Conv2D(filters=512,kernel_size=(3,3),padding="same")(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[13],include_quantization=False)
		x = MaxPool2D(pool_size=(2,2),strides=(2,2))(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[14],include_quantization=False)
		x = Conv2D(filters=512,kernel_size=(3,3),padding="same")(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[15],include_quantization=False)
		x = Conv2D(filters=512,kernel_size=(3,3),padding="same")(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[16],include_quantization=False)
		x = Conv2D(filters=512,kernel_size=(3,3),padding="same")(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[17],include_quantization=False)
		x = MaxPool2D(pool_size=(2,2),strides=(2,2))(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[18],include_quantization=False)
		x = Flatten()(x)
		x = Dense(4096)(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[19],include_quantization=False)
		x = Dense(4096)(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[20],include_quantization=False)
		x = Dense(output_shape)(x)
		x = AddCustomLayers(x,include_aging=False)
		x = tf.keras.activations.softmax(x)
		x = AddCustomLayers(x,include_aging=False)
		Net = tf.keras.Model(inputs=input_layer, outputs=x)
		return Net

	elif architecture == 'PilotNet':
		input_layer = tf.keras.Input(input_shape)
		x = Cropping2D(cropping=((50,20), (0,0)))(input_layer)
		x = Lambda(preprocess)(x)
		x = Lambda(lambda x: (x/ 127.0 - 1.0))(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[0])
		x = Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2))(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[1])
		x = Conv2D(filters=36, kernel_size=(5, 5),strides=(2, 2))(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[2])
		x = Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2))(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[3])
		x = Conv2D(filters=64, kernel_size=(3, 3))(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[4])
		x = Conv2D(filters=64, kernel_size=(3, 3))(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[5])
		x = Dropout(0.5)(x)
		x = Flatten()(x)
		x = Dense(units=1164)(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[6])
		x = Dense(units=100)(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[7])
		x = Dense(units=50)(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[8])
		x = Dense(units=10)(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[9])
		x = Dense(units=output_shape)(x)
		x = AddCustomLayers(x,include_aging=False)
		Net = tf.keras.Model(inputs=input_layer, outputs=x)
		return Net

	elif architecture == 'MobileNet':
		def MobilNetInitialConvBlock(inputs, filters, kernel=(3, 3), strides=(1, 1)):
			x = AddCustomLayers(inputs,include_aging=True,aging_active = aging_active[0])
			x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
			x = Conv2D(filters, kernel, padding='same', use_bias=False, strides=strides)(x)
			x = AddCustomLayers(x,include_aging=False)
			x = BatchNormalization()(x)
			x = ReLU(6.)(x)
			x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[1])
			return x

		def DepthwiseConvBlock(inputs, filters, strides=(1, 1), blockId=1):
			pad = 'same' if strides == (1, 1) else 'valid'
			x = inputs   if strides == (1, 1) else ZeroPadding2D(((0, 1), (0, 1)))(inputs)
			x = DepthwiseConv2D((3, 3), padding=pad, strides=strides, use_bias=False)(x)
			x = AddCustomLayers(x,include_aging=False)
			x = BatchNormalization()(x)
			x = ReLU(6.)(x)
			x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[2*blockId])
			x = Conv2D(filters, (1, 1), padding='same', strides=(1, 1), use_bias=False)(x)
			x = AddCustomLayers(x,include_aging=False)
			x = BatchNormalization()(x)
			x = ReLU(6.)(x)
			x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[2*blockId+1])
			return x

		input_layer = tf.keras.Input(input_shape)
		x = MobilNetInitialConvBlock(input_layer, 32, strides=(2, 2))
		x = DepthwiseConvBlock(x, 64,   blockId=1)
		x = DepthwiseConvBlock(x, 128,  blockId=2, strides=(2, 2))
		x = DepthwiseConvBlock(x, 128,  blockId=3)
		x = DepthwiseConvBlock(x, 256,  blockId=4, strides=(2, 2))
		x = DepthwiseConvBlock(x, 256,  blockId=5)
		x = DepthwiseConvBlock(x, 512,  blockId=6, strides=(2, 2))
		x = DepthwiseConvBlock(x, 512,  blockId=7)
		x = DepthwiseConvBlock(x, 512,  blockId=8)
		x = DepthwiseConvBlock(x, 512,  blockId=9)
		x = DepthwiseConvBlock(x, 512,  blockId=10)
		x = DepthwiseConvBlock(x, 512,  blockId=11)
		x = DepthwiseConvBlock(x, 1024, blockId=12, strides=(2, 2))
		x = DepthwiseConvBlock(x, 1024, blockId=13)
		x = GlobalAveragePooling2D()(x)
		x = Reshape((1, 1, 1024))(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[28])
		x = Dropout(1e-3)(x)
		x = Conv2D(output_shape, (1, 1), padding='same')(x)
		x = AddCustomLayers(x,include_aging=False)
		x = Reshape((output_shape,))(x)
		x = Activation(activation='softmax')(x)
		x = AddCustomLayers(x,include_aging=False)
		Net = tf.keras.Model(inputs=input_layer, outputs=x)
		return Net

	elif architecture == 'ZFNet':
		input_layer = tf.keras.Input(input_shape)
		x = AddCustomLayers(input_layer,include_aging=True,aging_active = aging_active[0])
		x = Conv2D(filters=96, kernel_size=(7,7), strides=(2,2), padding = 'valid')(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[1])
		x = MaxPool2D(pool_size=(3,3),strides=(2,2))(x)
		x = Lambda(lambda x: tf.image.per_image_standardization(x))(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[2])
		x = Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), padding = 'same')(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[3])
		x = MaxPool2D(pool_size=(3,3),strides=(2,2))(x)
		x = Lambda(lambda x: tf.image.per_image_standardization(x))(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[4])
		x = Conv2D(filters = 384, kernel_size=(3,3), padding='same')(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[5])
		x = Conv2D(filters = 384, kernel_size=(3,3), padding='same')(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[6])
		x = Conv2D(filters = 256, kernel_size=(3,3), padding='same')(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[7])
		x = MaxPool2D(pool_size=(3,3),strides=(2,2))(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[8],include_quantization=False)
		x = Flatten()(x)
		x = Dense(4096)(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[9])
		x = Dense(4096)(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[10])
		x = Dense(output_shape)(x)
		x = AddCustomLayers(x,include_aging=False)
		x = Activation(activation='softmax')(x)
		x = AddCustomLayers(x,include_aging=False)
		Net = tf.keras.Model(inputs=input_layer, outputs=x)
		return Net

	elif architecture == 'SqueezeNet':

		def FireBlock(inputs, fs, fe, blockId):
			#Squeeze
			s  = Conv2D(fs, 1)(inputs)
			s  = ReLU()(s)

			s  = AddCustomLayers(s,include_aging=True,aging_active = aging_active[blockId])
			#Expand
			e1 = Conv2D(fe, 1)(s)
			e1 = ReLU()(e1)
			e3 = Conv2D(fe, 3, padding = 'same')(s)
			e3 = ReLU()(e3)
			e  = Concatenate()([e1,e3])
			e = AddCustomLayers(e,include_aging=True,aging_active = aging_active[blockId+1])
			return e

		input_layer = tf.keras.Input(input_shape)
		x = AddCustomLayers(input_layer,include_aging=True,aging_active = aging_active[0])
		x = Conv2D(96,7,2,'same')(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[1])
		x = MaxPool2D(3,2,'same')(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[2], include_quantization=False)
		x = FireBlock(x, 16, 64,  3)
		x = FireBlock(x, 16, 64,  5)
		x = FireBlock(x, 32, 128, 7)
		x = MaxPool2D(3,2,'same')(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[9], include_quantization=False)
		x = FireBlock(x, 32, 128, 10)
		x = FireBlock(x, 48, 192, 12)
		x = FireBlock(x, 48, 192, 14)
		x = FireBlock(x, 64, 256, 16)
		x = MaxPool2D(3,2,'same')(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[18], include_quantization=False)
		x = FireBlock(x, 64, 256, 19)
		x = Conv2D(output_shape,1,(1,1),'same')(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[20])
		x = GlobalAveragePooling2D()(x)
		x = AddCustomLayers(x,include_aging=False)
		x = tf.keras.activations.softmax(x)
		x = AddCustomLayers(x,include_aging=False)
		Net = tf.keras.Model(inputs=input_layer, outputs=x)
		return Net

	elif architecture == 'SentimentalNet':
		top_words = 5000
		max_words = 500
		input_layer = tf.keras.Input(input_shape)
		x = Embedding(top_words, 32, input_length=max_words)(input_layer)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[0])
		x = Conv1D(filters=32, kernel_size=3, padding='same')(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[1])
		x = MaxPooling1D(pool_size=2)(x)
		#x = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[2])
		x = Flatten()(x)
		x = Dense(250)(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[3])
		x = Dense(output_shape)(x)
		x = AddCustomLayers(x,include_aging=False)
		x = tf.keras.activations.sigmoid(x)
		x = AddCustomLayers(x,include_aging=False)
		Net = tf.keras.Model(inputs=input_layer, outputs=x)
		return Net

	elif architecture == 'DenseNet':

		def ConvBlock(inputs, growthRate, blockId):
			x = BatchNormalization(epsilon=1.001e-5)(inputs)
			x = ReLU()(x)
			x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[blockId])
			x = Conv2D(4*growthRate, 1, use_bias=False)(x)
			x = AddCustomLayers(x,include_aging=False)
			x = BatchNormalization(epsilon=1.001e-5)(x)
			x = ReLU()(x)
			x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[blockId+1])
			x = Conv2D(growthRate, 3, padding='same', use_bias=False)(x)
			x = AddCustomLayers(x,include_aging=False)
			o = Concatenate()([inputs, x])
			return o

		def DenseBlock(x, blocks, blockId):
			for i in range(blocks):
				x = ConvBlock(x, 32, blockId+2*i)
			return x

		def TransitionBlock(x, blockId):
			x = BatchNormalization(epsilon=1.001e-5)(x)
			x = ReLU()(x)
			x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[blockId])
			x = Conv2D(int(K.int_shape(x)[-1] * 0.5), 1, use_bias=False)(x)
			x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[blockId+1])
			x = AveragePooling2D(2, strides=2)(x)
			x = AddCustomLayers(x,include_aging=False)
			return x

		input_layer = tf.keras.Input(input_shape)
		x = AddCustomLayers(input_layer,include_aging=True,aging_active = aging_active[0])
		x = ZeroPadding2D(padding=((3, 3), (3, 3)))(x)
		x = Conv2D(64, 7, strides = 2, use_bias = False)(x)
		x = AddCustomLayers(x,include_aging=False)
		x = BatchNormalization(epsilon=1.001e-5)(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[1])
		x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
		x = MaxPool2D(3, strides=2)(x)
		x = DenseBlock(x, 6, 2)
		x = TransitionBlock(x, 14)
		x = DenseBlock(x, 12, 16)
		x = TransitionBlock(x, 40)
		x = DenseBlock(x, 24,  42)
		x = TransitionBlock(x, 90)
		x = DenseBlock(x, 16,  92)
		x = BatchNormalization(epsilon=1.001e-5)(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[124])
		x = GlobalAveragePooling2D()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[125])
		x = Dense(output_shape)(x)
		x = AddCustomLayers(x,include_aging=False)
		x = tf.keras.activations.softmax(x)
		x = AddCustomLayers(x,include_aging=False)
		Net = tf.keras.Model(inputs=input_layer, outputs=x)
		return Net

	elif architecture == 'ResNet50':

		def res_identity(x, filters):
			''' renet block where dimension doesnot change.
            The skip connection is just simple identity conncection
            we will have 3 blocks and then input will be added
            '''
			x_skip = x  # this will be used for addition with the residual block
			f1, f2 = filters

			# first block
			x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
			x = BatchNormalization()(x)
			x = Activation(activations.relu)(x)
			x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[4], include_quantization=False)

			# second block # bottleneck (but size kept same with padding)
			x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
			x = BatchNormalization()(x)
			x = Activation(activations.relu)(x)
			x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[5], include_quantization=False)

			# third block activation used after adding the input
			x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
			x = BatchNormalization()(x)
			x = Activation(activations.relu)(x)
			x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[6], include_quantization=False)

			# add the input
			x = Add()([x, x_skip])
			x = Activation(activations.relu)(x)
			x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[7], include_quantization=False)

			return x

		# In[ ]:

		def res_conv(x, s, filters):

			# here the input size changes, when it goes via conv blocks  so the skip connection uses a projection (conv layer) matrix  '''
			x_skip = x
			f1, f2 = filters

			# first block
			x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x)
			# when s = 2 then it is like downsizing the feature map
			x = BatchNormalization()(x)
			x = Activation(activations.relu)(x)
			x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[8])

			# second block
			x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
			x = BatchNormalization()(x)
			x = Activation(activations.relu)(x)
			x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[9])

			# third block
			x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
			x = BatchNormalization()(x)

			# shortcut
			x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(
				x_skip)
			x_skip = BatchNormalization()(x_skip)

			# add
			x = Add()([x, x_skip])
			x = Activation(activations.relu)(x)
			x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[10])

			return x

		# In[ ]:

		#class_types = 8

		### Combine the above functions to build 50 layers resnet.
		#def resnet50(input_shape):

		input_layer = tf.keras.Input(input_shape)
		x = AddCustomLayers(input_layer, include_aging=True, aging_active=aging_active[0])
		x = ZeroPadding2D(padding=(3, 3))(input_layer)
		# 1st stage
		# here we perform maxpooling, see the figure above
		x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
		#   x = AddCustomLayers(x,include_aging=False)
		x = BatchNormalization()(x)
		x = Activation(activations.relu)(x)
		x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[1])
		x = MaxPooling2D((3, 3), strides=(2, 2))(x)

		# 2nd stage
		# frm here on only conv block and identity block, no pooling

		x = res_conv(x, s=1, filters=(64, 256))
		x = res_identity(x, filters=(64, 256))
		x = res_identity(x, filters=(64, 256))

		# 3rd stage

		x = res_conv(x, s=2, filters=(128, 512))
		x = res_identity(x, filters=(128, 512))
		x = res_identity(x, filters=(128, 512))
		x = res_identity(x, filters=(128, 512))

		# 4th stage

		x = res_conv(x, s=2, filters=(256, 1024))
		x = res_identity(x, filters=(256, 1024))
		x = res_identity(x, filters=(256, 1024))
		x = res_identity(x, filters=(256, 1024))
		x = res_identity(x, filters=(256, 1024))
		x = res_identity(x, filters=(256, 1024))

		# 5th stage

		x = res_conv(x, s=2, filters=(512, 2048))
		x = res_identity(x, filters=(512, 2048))
		x = res_identity(x, filters=(512, 2048))

		# ends with average pooling and dense connection

		x = AveragePooling2D((2, 2), padding='same')(x)
		x = Flatten()(x)
		x = Dense(496)(x)
		x = AddCustomLayers(x, include_aging=False)
		x = Activation(activations.relu)(x)
		x = Dropout(0.3)(x)
		x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[2], include_quantization=False)
		x = Dense(496)(x)
		x = AddCustomLayers(x, include_aging=False)
		x = Activation(activations.relu)(x)
		x = Dropout(0.3)(x)
		x= AddCustomLayers(x, include_aging=True, aging_active=aging_active[3], include_quantization=False)
		x = Dense(output_shape)(x)
		x = AddCustomLayers(x, include_aging=False)
		x = Activation(activation='softmax')(x)
		x = AddCustomLayers(x, include_aging=False)
		Net = tf.keras.Model(inputs=input_layer, outputs=x, name='Resnet50')
		return Net

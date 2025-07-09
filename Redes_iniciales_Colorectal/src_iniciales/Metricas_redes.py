#!/usr/bin/env python
# coding: utf-8

# # Accesos al buffer: Lecturas y Escrituras

# In[1]:



import os
import pickle as pickle
import tensorflow as tf
import numpy as np
from Nets_original  import GetNeuralNetworkModel
from Stats import WeightQuantization, ActivationStats, CheckAccuracyAndLoss, QuantizationEffect, GetReadAndWrites
from Training import GetDatasets, GetPilotNetDataset
import pandas as pd
from datetime import datetime




# Con el siguiente bloque obtiene  el número de lecturas y escrituras por posición de memoria tanto usando los experimentos sin usarlos

# In[2]:


from Simulation import buffer_simulation, save_obj, load_obj
locs=load_obj('Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/locs_054')
error_mask=load_obj('Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x//vc_707/error_mask_054')
#error_mask=error_mask[0:10]
print(error_mask[0:10])
print(locs[0:10])
#error_mask=error_mask[9000:9010]
#locs=locs[0:10]
print(len(locs))
print(len(error_mask))


# # AlexNet

# In[3]:


#
# word_size  = 16
# afrac_size = 11
# aint_size  = 4
# wfrac_size = 11
# wint_size  = 4
# trainBatchSize = testBatchSize = 1
# _, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (227, 227), 8, trainBatchSize, testBatchSize)
#
#
# cwd = os.getcwd()
# wgt_dir = os.path.join(cwd, 'Data')
# wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
# wgt_dir = os.path.join(wgt_dir, 'AlexNet')
# wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
# wgt_dir = os.path.join(wgt_dir,'Weights')
#
# #
# # # In[4]:
# #
# #
# # def Graficar(Experiment_Acceses):
# #
# #         plt.figure(figsize=(15, 5))
# #         plt.stackplot(Experiment_Acceses.to_dict()['index'].values(),
# #                       Experiment_Acceses.to_dict()['Lecturas'].values(),
# #                       Experiment_Acceses.to_dict()['Escrituras'].values(),
# #                       colors=['blue', 'orange'])
# #         plt.legend(['Reads','Writes'])
# #
# #
# # # In[5]:
# #
# #
# activation_aging = [True]*11
# AlexNet = GetNeuralNetworkModel('AlexNet', (227,227,3), 8, faulty_addresses=locs, masked_faults=error_mask,
#                                  aging_active=activation_aging, word_size=word_size, frac_size=afrac_size,
#                                  batch_size = testBatchSize)
# AlexNet.load_weights(wgt_dir).expect_partial()
# loss = tf.keras.losses.CategoricalCrossentropy()
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# WeightQuantization(model=AlexNet, frac_bits=wfrac_size, int_bits=wint_size)
# AlexNet.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
# #loss,acc =AlexNet.evaluate(test_dataset)
# #
# #
# # # In[7]:
# #
# #
# #
# num_address  =1048576
# VBW=[101102, 124520, 302065, 331634, 340492, 340493, 360087, 360088, 377722, 377723, 419841, 458633, 458634, 465007, 465008, 465034, 465197, 465389, 544769, 544770, 545758, 590341, 590402, 590403, 590404, 590405, 590406, 590407, 590408, 590409, 590424, 590610, 590706, 590707, 590790, 590804, 611329]  # Tamaño del buffer, igual a la capa mas grande de la red (este caso), o un tamaño fijo pre-establecido.
# Indices      = [0,3,9,11,17,19,25,31,37,40,45,50] #Capas con la informacion de procesamiento
# samples      = 150 #Numero de imagenes
# # Sin Power Gating:
# Data         = GetReadAndWrites(AlexNet,Indices,num_address,samples,CNN_gating=False,network_name='AlexNet')
# stats        = {'Lecturas': Data['Reads'],'Escrituras': Data['Writes']}
# Baseline_Acceses   = pd.DataFrame(stats).reset_index(drop=False)
# # Con Power Gating
# # Data    = GetReadAndWrites(AlexNet,Indices,num_address,samples,CNN_gating=True)
# # stats    = {'Lecturas': Data['Reads'],'Escrituras': Data['Writes']}
# Experiment_Acceses = pd.DataFrame(stats).reset_index(drop=False)
# df_writes_Read =  pd.concat([Baseline_Acceses], axis=1, join='outer')
# df_writes_Read .columns = ['index','Lecturas','Escrituras']
#
# with pd.ExcelWriter('Analizando_fichero_detalle/Alterado_fichero/AlexNet_reads_and_write_num_adress_total.xlsx') as writer:
#        df_writes_Read.to_excel(writer, sheet_name='base', index=False)
# save_obj(Baseline_Acceses,'Data/Acceses/AlexNet/Baseline')
#save_obj(Experiment_Acceses,'Data/Acceses/AlexNet/Experiment')



# In[13]:


#Graficar(Baseline_Acceses)


# In[12]:


#Graficar(Experiment_Acceses)


# Gráficar

# Accesos a la cache: durante el proceso de inferencia VBW

# In[9]:


# VBW=[101102, 124520, 302065, 331634, 340492, 340493, 360087, 360088, 377722, 377723, 419841, 458633, 458634, 465007, 465008, 465034, 465197, 465389, 544769, 544770, 545758, 590341, 590402, 590403, 590404, 590405, 590406, 590407, 590408, 590409, 590424, 590610, 590706, 590707, 590790, 590804, 611329]  # Tamaño del buffer, igual a la capa mas grande de la red (este caso), o un tamaño fijo pre-establecido.
# writes_Read_VBW=df_writes_Read.iloc[VBW]
# with pd.ExcelWriter('Analizando_fichero_detalle/Alterado_fichero/Lectura_escritura_buffer/AlexNet_reads_and_write_VBW.xlsx') as writer:
#         writes_Read_VBW.to_excel(writer, sheet_name='base', index=False)
#


# ciclo de jecución de la red

# In[10]:



# LI = [0,3,9,11,17,19,25,31,37,40,45,50]
# AI = [2,8,10,16,18,24,30,36,38,44,49,53]
# Buffer,ciclos =  buffer_simulation(AlexNet, test_dataset, integer_bits = 4, fractional_bits = 11, samples = 5, start_from = 0,
#                                   bit_invertion = False, bit_shifting = False, CNN_gating = False,
#                                   buffer_size = 1048576, write_mode ='default', save_results = True,
#                                   results_dir = 'Data/Stats/AlexNet/Colorectal Dataset/CNN-Gated/',
#                                   layer_indexes = LI , activation_indixes = AI)


# SqueezeNet




word_size  = 16
afrac_size = 9
aint_size  = 6
wfrac_size = 15
wint_size  = 0

trainBatchSize = testBatchSize = 1
_, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)


cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'SqueezeNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir,'Weights')


# In[4]:


activation_aging = [True]*22


#Acá la creamos, notese que como no se introduciran fallos en activaciones no es necesario pasar locs ni masks
SqueezeNet = GetNeuralNetworkModel('SqueezeNet', (224,224,3), 8, faulty_addresses=locs, masked_faults=error_mask,
                                 aging_active=activation_aging, word_size=word_size, frac_size=afrac_size,
                                 batch_size = testBatchSize)
SqueezeNet.load_weights(wgt_dir).expect_partial()
loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
SqueezeNet.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
WeightQuantization(model=SqueezeNet, frac_bits=wfrac_size, int_bits=wint_size)
#loss,acc =SqueezeNet.evaluate(test_dataset)

#

#
num_address  =1048576
samples      = 150
Indices = [0,3,7, 9,(13,14),20,(24,25),31,(35,36),42,44,(48,49),55,(59,60),66,(70,71),77,(81,82),88,90,(94,95),101,104]
Data    = GetReadAndWrites(SqueezeNet,Indices,num_address,samples,CNN_gating=False,network_name='SqueezeNet')
stats   = {'Lecturas': Data['Reads'],'Escrituras': Data['Writes']}
Baseline_Acceses   = pd.DataFrame(stats).reset_index(drop=False)
Data    = GetReadAndWrites(SqueezeNet,Indices,num_address,samples,CNN_gating=True,network_name='SqueezeNet')
stats    = {'Lecturas': Data['Reads'],'Escrituras': Data['Writes']}
Experiment_Acceses = pd.DataFrame(stats).reset_index(drop=False)
df_writes_Read =  pd.concat([Baseline_Acceses], axis=1, join='outer')
df_writes_Read.columns = ['index','Lecturas','Escrituras']
with pd.ExcelWriter('Analizando_fichero_detalle/Alterado_fichero/SqueezeNet_reads_and_write_num_adress_total.xlsx') as writer:
       df_writes_Read.to_excel(writer, sheet_name='base', index=False)
##
save_obj(Baseline_Acceses,'Data/Acceses/SqueezeNet/Baseline_ya')
#save_obj(Experiment_Acceses,'Data/Acceses/SqueezeNet/Experiment')


# In[ ]:


#Graficar(Baseline_Acceses)


# In[ ]:


#Graficar(Experiment_Acceses)


# In[ ]:


VBW=[101102, 124520, 302065, 331634, 340492, 340493, 360087, 360088, 377722, 377723, 419841, 458633, 458634, 465007, 465008, 465034, 465197, 465389, 544769, 544770, 545758, 590341, 590402, 590403, 590404, 590405, 590406, 590407, 590408, 590409, 590424, 590610, 590706, 590707, 590790, 590804, 611329]  # Tamaño del buffer, igual a la capa mas grande de la red (este caso), o un tamaño fijo pre-establecido.
writes_Read_VBW=df_writes_Read.iloc[VBW]
with pd.ExcelWriter('Analizando_fichero_detalle/Alterado_fichero/Lectura_escritura_buffer/SqueezeNett_reads_and_write_VBW.xlsx') as writer:
        writes_Read_VBW.to_excel(writer, sheet_name='base', index=False)
##


# ciclos

# In[ ]:




LI = [0,3,7, 9,(13,14),20,(24,25),31,(35,36),42,44,(48,49),55,(59,60),66,(70,71),77,(81,82),88,90,(94,95),101,104]
AI = [2,6,8,12,     19,23,     30,34,     41,43,47,     54,58,     65,69,     76,80 ,    87,89,93,    100,103,107]
buffer_simulation(SqueezeNet,test_dataset, integer_bits = 6, fractional_bits = 9, samples = 150, start_from = 0,
                 bit_invertion = False, bit_shifting = False, CNN_gating = False, write_mode ='default',
                 results_dir = 'Data/Stats/SqueezeNet/', buffer_size = num_address,
                 layer_indexes = LI , activation_indixes = AI)


# DenseNet




trainBatchSize = testBatchSize = 1
_, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)

word_size  = 16
afrac_size = 12
aint_size  = 3
wfrac_size = 13
wint_size  = 2

# Tamaño del buffer de activaciones == al tamaño de la capa con mayor numero de activaciones (290400 pesos de 16 bits cada uno)
#abuffer_size = 16777216
# Directorio de los pesos
cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'DenseNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir,'Weights')


# In[4]:


activation_aging = [True]*188

DenseNet = GetNeuralNetworkModel('DenseNet', (224,224,3), 8, faulty_addresses=locs, masked_faults=error_mask,
                                 aging_active=activation_aging, word_size=word_size, frac_size=afrac_size,
                                 batch_size = testBatchSize)
DenseNet.load_weights(wgt_dir).expect_partial()
loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
DenseNet.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
WeightQuantization(model=DenseNet, frac_bits=wfrac_size, int_bits=wint_size)
#loss,acc =DenseNet.evaluate(test_dataset)
#

# In[ ]:



num_address  =1048576
samples      = 150
Indices=[0,4,11,12,16,(22,11),25,29,(35,24),38,42,(48,37),51,55,(61,50),64,68,(74,63),77,81,(87,76),90,94,97,99,103,(109,97),
        112,116,(122,111),125,129,(135,124),138,142,(148,137),151,155,(161,150),164,168,(174,163),177,181,(187,176),
        190,194,(200,189),203,207,(213,202),216,220,(226,215),229,233,(239,228),242,246,(252,241),255,259,262,264,268,(274,262),
        277,281,(287,276),290,294,(300,289),303,307,(313,302),316,320,(326,315),329,333,(339,328),342,346,(352,341),
        355,359,(365,354),368,372,(378,367),381,385,(391,380),394,398,(404,393),407,411,(417,406),420,424,(430,419),
        433,437,(443,432),446,450,(456,445),459,463,(469,458),472,476,(482,471),485,489,(495,484),498,502,(508,497),
        511,515,(521,510),524,528,(534,523),537,541,(547,536),550,554,(560,549),563,567,(573,562),576,580,583,585,589,(595,583),
        598,602,(608,597),611,615,(621,610),624,628,(634,623),637,641,(647,636),650,654,(660,649),663,667,(673,662),
        676,680,(686,675),689,693,(699,688),702,706,(712,701),715,719,(725,714),728,732,(738,727),741,745,(751,740),
        754,758,(764,753),767,771,(777,765),780,784,(790,779),793,797,800]
Data     = GetReadAndWrites(DenseNet,Indices,num_address,samples,CNN_gating=False,network_name='DenseNet')
stats    = {'Lecturas': Data['Reads'],'Escrituras': Data['Writes']}
Baseline_Acceses   = pd.DataFrame(stats).reset_index(drop=False)
# Data     = GetReadAndWrites(DenseNet,Indices,num_address,samples,CNN_gating=True,network_name='DenseNet')
# stats    = {'Lecturas': Data['Reads'],'Escrituras': Data['Writes']}

Experiment_Acceses = pd.DataFrame(stats).reset_index(drop=False)
df_writes_Read =  pd.concat([Baseline_Acceses], axis=1, join='outer')
df_writes_Read.columns = ['index','Lecturas','Escrituras']
with pd.ExcelWriter('Analizando_fichero_detalle/Alterado_fichero/Lectura_escritura_buffer/DenseNet_reads_and_write_num_adress_total.xlsx') as writer:
        df_writes_Read.to_excel(writer, sheet_name='base', index=False)
save_obj(Baseline_Acceses,'Data/Acceses/DenseNet/Baseline')
#save_obj(Experiment_Acceses,'Data/Acceses/DenseNet/Experiment')


# In[ ]:


#Graficar(Baseline_Acceses)


# In[ ]:


#Graficar(Experiment_Acceses)


# In[ ]:


VBW=[101102, 124520, 302065, 331634, 340492, 340493, 360087, 360088, 377722, 377723, 419841, 458633, 458634, 465007, 465008, 465034, 465197, 465389, 544769, 544770, 545758, 590341, 590402, 590403, 590404, 590405, 590406, 590407, 590408, 590409, 590424, 590610, 590706, 590707, 590790, 590804, 611329]  # Tamaño del buffer, igual a la capa mas grande de la red (este caso), o un tamaño fijo pre-establecido.
writes_Read_VBW=df_writes_Read.iloc[VBW]
with pd.ExcelWriter('Analizando_fichero_detalle/Alterado_fichero/Lectura_escritura_buffer/DenseNet_reads_and_write_VBW.xlsx') as writer:
        writes_Read_VBW.to_excel(writer, sheet_name='base', index=False)


# ciclos : analizar por qué start_fron esta en 137 correo con 0  aver que pasa

# In[ ]:


LI = [0,4,11,12,16,     22,25,29,     35,38,42,     48,51,55,     61,64,68,     74,77,81,     87,90,94,97,99,103,      109,
     112,116,      122,125,129,      135,138,142,      148,151,155,      161,164,168,      174,177,181,      187,
     190,194,      200,203,207,      213,216,220,      226,229,233,      239,242,246,      252,255,259,262,264,268,      274,
     277,281,      287,290,294,      300,303,307,      313,316,320,      326,329,333,      339,342,346,      352,
     355,359,      365,368,372,      378,381,385,      391,394,398,      404,407,411,      417,420,424,      430,
     433,437,      443,446,450,      456,459,463,      469,472,476,      482,485,489,      495,498,502,      508,
     511,515,      521,524,528,      534,537,541,      547,550,554,      560,563,567,      573,576,580,583,585,589,      595,
     598,602,      608,611,615,      621,624,628,      634,637,641,      647,650,654,      660,663,667,      673,
     676,680,      686,689,693,      699,702,706,      712,715,719,      725,728,732,      738,741,745,      751,
     754,758,      764,767,771,      777,780,784,      790,793,797,800]
AI = [2,9,11,15,21,(23,11),28,34,(36,24),41,47,(49,37),54,60,(62,50),67,73,(75,63),80,86,(88,76),93,96,98,102,108,(110,98),
     115,121,(123,111),128,134,(136,124),141,147,(149,137),154,160,(162,150),167,173,(175,163),180,186,(188,176),
     193,199,(201,189),206,212,(214,202),219,225,(227,215),232,238,(240,228),245,251,(253,241),258,261,263,267,273,(275,263),
     280,286,(288,276),293,299,(301,289),306,312,(314,302),319,325,(327,315),332,338,(340,328),345,351,(353,341),
     358,364,(366,353),371,377,(379,367),384,390,(392,380),397,403,(405,393),410,416,(418,406),423,429,(431,419),
     436,442,(444,432),449,455,(457,445),462,468,(470,458),475,481,(483,471),488,494,(496,484),501,507,(509,497),
     514,520,(522,510),527,533,(535,523),540,546,(548,536),553,559,(561,549),566,572,(574,562),579,582,584,588,594,(596,584),
     601,607,(609,597),614,620,(622,610),627,633,(635,623),640,646,(648,636),653,659,(661,649),666,672,(674,662),
     679,685,(687,675),692,698,(700,688),705,711,(713,701),718,724,(726,714),731,737,(739,727),744,750,(752,740),
     757,763,(765,753),770,776,(778,766),783,789,(791,779),796,799,803]
buffer_simulation(DenseNet,test_dataset, integer_bits = 3, fractional_bits = 12, samples = 150, start_from = 0,
                 bit_invertion = False, bit_shifting = False, CNN_gating = False, write_mode ='default',
                 results_dir = 'Data/Stats/DenseNet/', buffer_size = num_address,
                 layer_indexes = LI , activation_indixes = AI)


#
# # MobileNet

# In[ ]:


trainBatchSize = testBatchSize = 1
_, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)


# In[3]:


#Numero de bits para activaciones (a) y pesos (w)
word_size  = 16
afrac_size = 11
aint_size  = 4
wfrac_size = 14
wint_size  = 1

# Tamaño del buffer de activaciones == al tamaño de la capa con mayor numero de activaciones (290400 pesos de 16 bits cada uno)
#abuffer_size = 16777216
# Directorio de los pesos
cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'MobileNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir,'Weights')


# In[4]:




# In[ ]:


activation_aging = [True]*29
MobileNet = GetNeuralNetworkModel('MobileNet', (224,224,3), 8, faulty_addresses=locs, masked_faults=error_mask,
                                 aging_active=activation_aging, word_size=word_size, frac_size=afrac_size,  batch_size = testBatchSize)
MobileNet.load_weights(wgt_dir).expect_partial()
loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
MobileNet.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
WeightQuantization(model=MobileNet, frac_bits=wfrac_size, int_bits=wint_size)
#loss,acc  = MobileNet.evaluate(test_dataset)






#
#
# # In[ ]:
#
#
num_address  =1048576
samples      = 150
Indices = [0,4,10,16,23,29,35,41,48,54,60,66,73,79,85,91,97,103,109,115,121,127,133,139,146,152,158,164,170,175]
Data    = GetReadAndWrites(MobileNet,Indices,num_address,samples,CNN_gating=False, network_name='MobileNet')
stats   = {'Lecturas': Data['Reads'],'Escrituras': Data['Writes']}
Baseline_Acceses   = pd.DataFrame(stats).reset_index(drop=False)
Data    = GetReadAndWrites(MobileNet,Indices,num_address,samples,CNN_gating=True)
stats    = {'Lecturas': Data['Reads'],'Escrituras': Data['Writes']}
Experiment_Acceses = pd.DataFrame(stats).reset_index(drop=False)
df_writes_Read =  pd.concat([Baseline_Acceses], axis=1, join='outer')
df_writes_Read.columns = ['index','Lecturas','Escrituras']
with pd.ExcelWriter('Analizando_fichero_detalle/Alterado_fichero/Lectura_escritura_buffer/MobileNet_reads_and_write_num_adress_total.xlsx') as writer:
        df_writes_Read.to_excel(writer, sheet_name='base', index=False)
save_obj(Baseline_Acceses,'Data/Acceses/MobileNet/Baseline')
#save_obj(Experiment_Acceses,'Data/Acceses/MobileNet/Experiment')


# In[ ]:


#Graficar(Baseline_Acceses)


# In[ ]:


#Graficar(Experiment_Acceses)


# In[ ]:


VBW=[101102, 124520, 302065, 331634, 340492, 340493, 360087, 360088, 377722, 377723, 419841, 458633, 458634, 465007, 465008, 465034, 465197, 465389, 544769, 544770, 545758, 590341, 590402, 590403, 590404, 590405, 590406, 590407, 590408, 590409, 590424, 590610, 590706, 590707, 590790, 590804, 611329]  # Tamaño del buffer, igual a la capa mas grande de la red (este caso), o un tamaño fijo pre-establecido.
writes_Read_VBW=df_writes_Read.iloc[VBW]
with pd.ExcelWriter('Analizando_fichero_detalle/Alterado_fichero/Lectura_escritura_buffer/MobileNet_reads_and_write_VBW.xlsx') as writer:
        writes_Read_VBW.to_excel(writer, sheet_name='base', index=False)


# ciclos analizar start_from = 128

# In[ ]:


LI = [0,4,10,16,23,29,35,41,48,54,60,66,73,79,85,91,97,103,109,115,121,127,133,139,146,152,158,164,170,175]
AI = [2,9,15,21,28,34,40,46,53,59,65,71,78,84,90,96,102,108,114,120,126,132,138,144,151,157,163,169,173,179]
buffer_simulation(MobileNet,test_dataset, integer_bits = 4, fractional_bits = 11, samples = 150, start_from = 0,
                 bit_invertion = False, bit_shifting = False, CNN_gating = False, write_mode ='default',
                 results_dir = 'Data/Stats/MobileNet/', buffer_size = num_address,
                 layer_indexes = LI , activation_indixes = AI)

#
# # # VGG16

# In[ ]:


trainBatchSize = testBatchSize = 1
_, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)






word_size  = 16
afrac_size = 12
aint_size  = 3
wfrac_size = 15
wint_size  = 0

# Tamaño del buffer de activaciones == al tamaño de la capa con mayor numero de activaciones (290400 pesos de 16 bits cada uno)
#abuffer_size = 16777216
# Directorio de los pesos
cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'VGG16')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir,'Weights')


# In[4]:


activation_aging = [True]*21

VGG16 = GetNeuralNetworkModel('VGG16', (224,224,3), 8, faulty_addresses=locs, masked_faults=error_mask,aging_active=activation_aging, word_size=word_size, frac_size=afrac_size,
                                 batch_size = testBatchSize)
VGG16.load_weights(wgt_dir).expect_partial()
loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
VGG16.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
WeightQuantization(model=VGG16, frac_bits=wfrac_size, int_bits=wint_size)
#loss,acc  = VGG16.evaluate(test_dataset)
#
#
# # In[ ]:
#
#
num_address  =1048576
samples      = 150

Indices = [0,3,7,11,13,17,21,23,27,31,35,37,41,45,49,51,55,59,63,66,70,74]
Data    = GetReadAndWrites(VGG16,Indices,num_address,samples,CNN_gating=False, network_name='VGG16')
stats   = {'Lecturas': Data['Reads'],'Escrituras': Data['Writes']}
Baseline_Acceses   = pd.DataFrame(stats).reset_index(drop=False)
Data    = GetReadAndWrites(VGG16,Indices,num_address,samples,CNN_gating=True)
stats   = {'Lecturas': Data['Reads'],'Escrituras': Data['Writes']}
Experiment_Acceses = pd.DataFrame(stats).reset_index(drop=False)
#
df_writes_Read =  pd.concat([Baseline_Acceses], axis=1, join='outer')
df_writes_Read.columns = ['index','Lecturas','Escrituras']
#print(df_resumen_bit)
with pd.ExcelWriter('Analizando_fichero_detalle/Alterado_fichero/VGG16_reads_and_write_num_adress_tota.xlsx') as writer:
        df_writes_Read.to_excel(writer, sheet_name='base', index=False)
save_obj(Baseline_Acceses,'Data/Acceses/VGG16/Baseline')
# save_obj(Experiment_Acceses,'Data/Acceses/VGG16/Experiment')
#
#
# # In[ ]:
#
#
# #Graficar(Baseline_Acceses)
#
#
# # In[ ]:
#
#
# #Graficar(Experiment_Acceses)
#
#
# # In[ ]:
#
#
VBW=[101102, 124520, 302065, 331634, 340492, 340493, 360087, 360088, 377722, 377723, 419841, 458633, 458634, 465007, 465008, 465034, 465197, 465389, 544769, 544770, 545758, 590341, 590402, 590403, 590404, 590405, 590406, 590407, 590408, 590409, 590424, 590610, 590706, 590707, 590790, 590804, 611329]  # Tamaño del buffer, igual a la capa mas grande de la red (este caso), o un tamaño fijo pre-establecido.
writes_Read_VBW=df_writes_Read.iloc[VBW]
with pd.ExcelWriter('Analizando_fichero_detalle/Alterado_fichero/Lectura_escritura_buffer/VGG16_reads_and_write_VBW.xlsx') as writer:
        writes_Read_VBW.to_excel(writer, sheet_name='base', index=False)


# # In[ ]:
#
#
LI = [0,3,7,11,13,17,21,23,27,31,35,37,41,45,49,51,55,59,63,66,70,74]
AI = [2,6,10,12,16,20,22,26,30,34,36,40,44,48,50,54,58,62,64,69,73,77]
buffer_simulation(VGG16,test_dataset, integer_bits = 3, fractional_bits = 12, samples = 150, start_from = 0,
                 bit_invertion = False, bit_shifting = False, CNN_gating = False, write_mode ='default',
                 results_dir = 'Data/Stats/VGG16/', buffer_size = num_address,
                 layer_indexes = LI , activation_indixes = AI)


# # ZFNet

# In[ ]:


trainBatchSize = testBatchSize = 1
_, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)



word_size  = 16
afrac_size = 11
aint_size  = 4
wfrac_size = 14
wint_size  = 1


cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'ZFNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir,'Weights')


# In[4]:


activation_aging = [True]*11

ZFNet = GetNeuralNetworkModel('ZFNet', (224,224,3), 8, faulty_addresses=locs, masked_faults=error_mask,
                                 aging_active=activation_aging, word_size=word_size, frac_size=afrac_size,
                                 batch_size = testBatchSize)
ZFNet.load_weights(wgt_dir).expect_partial()
loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
ZFNet.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
WeightQuantization(model=ZFNet, frac_bits=wfrac_size, int_bits=wint_size)
#loss,acc  = ZFNet.evaluate(test_dataset)

#
# # In[ ]:
#
#
num_address  =1048576
samples      = 150
Indices = [0,3,7,11,15,19,23,27,31,34,37,40]
Data    = GetReadAndWrites(ZFNet,Indices,num_address,samples,CNN_gating=False, network_name='ZFNet')
stats   = {'Lecturas': Data['Reads'],'Escrituras': Data['Writes']}
Baseline_Acceses   = pd.DataFrame(stats).reset_index(drop=False)
# Data    = GetReadAndWrites(ZFNet,Indices,num_address,samples,CNN_gating=True)
# stats   = {'Lecturas': Data['Reads'],'Escrituras': Data['Writes']}
Experiment_Acceses = pd.DataFrame(stats).reset_index(drop=False)
df_writes_Read =  pd.concat([Baseline_Acceses], axis=1, join='outer')
df_writes_Read.columns = ['index','Lecturas','Escrituras']
#print(df_resumen_bit)
with pd.ExcelWriter('Analizando_fichero_detalle/Alterado_fichero/ZFNet_reads_and_write_num_adress_tota.xlsx') as writer:
        df_writes_Read.to_excel(writer, sheet_name='base', index=False)
save_obj(Baseline_Acceses,'Data/Acceses/ZFNet/Baseline')
#save_obj(Experiment_Acceses,'Data/Acceses/ZFNet/Experiment')


# In[ ]:


#Graficar(Baseline_Acceses)


# In[ ]:


#Graficar(Experiment_Acceses)


# In[ ]:


VBW=[101102, 124520, 302065, 331634, 340492, 340493, 360087, 360088, 377722, 377723, 419841, 458633, 458634, 465007, 465008, 465034, 465197, 465389, 544769, 544770, 545758, 590341, 590402, 590403, 590404, 590405, 590406, 590407, 590408, 590409, 590424, 590610, 590706, 590707, 590790, 590804, 611329]  # Tamaño del buffer, igual a la capa mas grande de la red (este caso), o un tamaño fijo pre-establecido.
writes_Read_VBW=df_writes_Read.iloc[VBW]
with pd.ExcelWriter('Analizando_fichero_detalle/Alterado_fichero/Lectura_escritura_buffer/VGG16_reads_and_write_VBW.xlsx') as writer:
        writes_Read_VBW.to_excel(writer, sheet_name='base', index=False)
#
#
# # ciclos
#
# # In[ ]:
#
#
LI = [0,3,7 ,11,15,19,23,27,31,34,37,40]
AI = [2,6,10,14,18,22,26,30,32,36,39,43]
buffer_simulation(ZFNet,test_dataset, integer_bits = 4, fractional_bits = 11, samples = 150, start_from = 0,
                 bit_invertion = False, bit_shifting = False, CNN_gating = False, write_mode ='default',
                 results_dir = 'Data/Stats/ZFNet/', buffer_size = num_address,
                 layer_indexes = LI , activation_indixes = AI)
#
#
# # # PiloNet
#
# # In[ ]:
#
#
trainBatchSize = testBatchSize = 1
_,_,test_dataset = GetPilotNetDataset(csv_dir='Data/Datos/driving_log.csv', train_batch_size=1, test_batch_size=1)



word_size  = 16
afrac_size = 15
aint_size  = 0
wfrac_size = 15
wint_size  = 0

# Directorio de los pesos
cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'PilotNet')
wgt_dir = os.path.join(wgt_dir,'Weights')



# In[4]:


activation_aging = [True]*10

PilotNet = GetNeuralNetworkModel('PilotNet', (160,320,3), 1, faulty_addresses=locs, masked_faults=error_mask,
                                 aging_active=activation_aging, word_size=word_size, frac_size=afrac_size,
                                 batch_size = testBatchSize)
PilotNet.load_weights(wgt_dir).expect_partial()
loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
PilotNet.compile(optimizer=optimizer, loss='mse')
WeightQuantization(model=PilotNet, frac_bits=wfrac_size, int_bits=wint_size)
#loss  = PilotNet.evaluate(test_dataset)

#
# # In[ ]:
#
#
num_address  =1048576
samples      = 150
Indices = [5,6,10,14,18,22,28,32,36,40,44]
Data    = GetReadAndWrites(PilotNet,Indices,num_address,samples,CNN_gating=False, network_name='PilotNet')
stats   = {'Lecturas': Data['Reads'],'Escrituras': Data['Writes']}
Baseline_Acceses   = pd.DataFrame(stats).reset_index(drop=False)
Data   = GetReadAndWrites(PilotNet,Indices,num_address,samples,CNN_gating=True)
stats    = {'Lecturas': Data['Reads'],'Escrituras': Data['Writes']}
Experiment_Acceses = pd.DataFrame(stats).reset_index(drop=False)
df_writes_Read =  pd.concat([Baseline_Acceses], axis=1, join='outer')
df_writes_Read.columns = ['index','Lecturas','Escrituras']
with pd.ExcelWriter('Analizando_fichero_detalle/Alterado_fichero/PilotNet_reads_and_write_num_adress_tota.xlsx') as writer:
        df_writes_Read.to_excel(writer, sheet_name='base', index=False)
save_obj(Baseline_Acceses,'Data/Acceses/PilotNet/Baseline')
#save_obj(Experiment_Acceses,'Data/Acceses/PilotNet/Experiment')


# In[ ]:
#
#
# #Graficar(Baseline_Acceses)
#
#
# # In[ ]:
#
#
# #Graficar(Experiment_Acceses)
#
#
# # In[ ]:
#
#
VBW=[101102, 124520, 302065, 331634, 340492, 340493, 360087, 360088, 377722, 377723, 419841, 458633, 458634, 465007, 465008, 465034, 465197, 465389, 544769, 544770, 545758, 590341, 590402, 590403, 590404, 590405, 590406, 590407, 590408, 590409, 590424, 590610, 590706, 590707, 590790, 590804, 611329]  # Tamaño del buffer, igual a la capa mas grande de la red (este caso), o un tamaño fijo pre-establecido.
writes_Read_VBW=df_writes_Read.iloc[VBW]
with pd.ExcelWriter('Analizando_fichero_detalle/Alterado_fichero/Lectura_escritura_buffer/PilotNet_reads_and_write_VBW.xlsx') as writer:
        writes_Read_VBW.to_excel(writer, sheet_name='base', index=False)


# Ciclos
#
# # In[ ]:
#
#
LI = [5,6,10,14,18,22,28,32,36,40,44]
AI = [5,9,13,17,21,25,31,35,39,43,45]
buffer_simulation(PilotNet,test_dataset, integer_bits = 0, fractional_bits = 15, samples = 150, start_from = 0,
                 bit_invertion = False, bit_shifting = False, CNN_gating = False, write_mode ='default',
                 results_dir = 'Data/Stats/PilotNet/', buffer_size = num_address,
                 layer_indexes = LI , activation_indixes = AI)
#
print(str()+' operación completada: ', datetime.now().strftime("%H:%M:%S"))
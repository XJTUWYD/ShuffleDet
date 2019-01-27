from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import pywrap_tensorflow
import os
import numpy as np
import cv2
from config.config import *
cfg = base_model_config()

def float_to_8bit(float_list):
	max_value = np.max(np.abs(float_list))
	if(max_value ==0):
		hex_list = [hex(0) for x in float_list]
		ratio = 0
	else:
		ratio = float(127/max_value)
		int_list = [int(np.round(x*ratio)) for x in float_list]
		hex_list = [ hex(x) if x>=0 else hex(-x+128) for x in int_list]
	return hex_list,ratio

def ckpt_split(file_path,layer_name,save_path):
  reader = pywrap_tensorflow.NewCheckpointReader(file_path)
  var_to_shape_map = reader.get_variable_to_shape_map()
  layer_path = save_path+'/'+layer_name
  ratio_key = open(save_path+'/ratio_key.txt','a')
  ratio_value = open(save_path+'/ratio_value.txt','a')
  for key in var_to_shape_map:
  	file_list = key.split('/')
  	if('Momentum' in key):
  		continue
  	else:
  		if(layer_name in file_list and 'kernels' in file_list and len(file_list)<=2):
  			weight = reader.get_tensor(key)
  			weight = list(np.ravel(weight))
  			hex_list,ratio = float_to_8bit(weight)
  			weight_file = open(layer_path+'_weight.txt','w+')
  			for i in range(len(hex_list)):
  				weight_file.write(str(hex_list[i])+'\n')
        		ratio_key.write(layer_name+'_weight'+'\n')
        		ratio_value.write(str(ratio)+'\n')
        		weight_file.close()
      		elif(layer_name in file_list and 'biases' in file_list):
        		bias = reader.get_tensor(key)
       			bias = list(np.ravel(bias))
       		 	hex_list,ratio = float_to_8bit(bias)
        		bias_file = open(layer_path+'_bias.txt','w+')
        		for i in range(len(hex_list)):
          			bias_file.write(str(hex_list[i])+'\n')
        		ratio_key.write(layer_name+'_bias'+'\n')
        		ratio_value.write(str(ratio)+'\n')
        		bias_file.close()
      		elif(layer_name in file_list and 'fold' in key):
        		quant = reader.get_tensor(key)
        		quant = list(np.ravel(quant))
        		hex_list,ratio = float_to_8bit(quant)
        		txt_name = layer_path+'_'+file_list[1]+'_'+file_list[2]+'.txt'
        		quant_file = open(txt_name,'w+')
        		for i in range(len(hex_list)):
          			quant_file.write(str(hex_list[i])+'\n')
        		ratio_key.write(layer_name+'_'+file_list[1]+'_'+file_list[2]+'\n')
        		ratio_value.write(str(ratio)+'\n')
        		quant_file.close()
  
im_path = cfg.IM_PATH
img = cv2.imread(im_path)
img = img.astype(np.float32, copy=False)
mean = np.array([[[103.939, 116.779, 123.68]]])
img -= mean
img = np.where(img<-127,-127,img)
img = np.where(img>127,127,img)
img = list(np.ravel(img))
hex_list,im_ratio = float_to_8bit(img)
LocalPath = os.getcwd()
input_save = LocalPath+cfg.SAVE_PATH+'/input'
if not os.path.exists(input_save):
  os.mkdir(input_save)
im_file = open(input_save+'/input.txt','w+')
for i in range(len(hex_list)):
  im_file.write(str(hex_list[i])+'\n')
im_file.close()

checkpoint_path = LocalPath+'/'+cfg.CHECKPOINT
parameter_save = LocalPath+cfg.SAVE_PATH+'/parameter'
if not os.path.exists(parameter_save):
  os.mkdir(parameter_save)
ratio_key = open(parameter_save+'/ratio_key.txt','w+')
ratio_value = open(parameter_save+'/ratio_value.txt','w+')
for x in cfg.LAYER_LIST:
  ckpt_split(checkpoint_path,x,parameter_save)
ratio_key.close()
ratio_value.close()


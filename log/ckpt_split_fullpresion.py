from tensorflow.python import pywrap_tensorflow
import os
import numpy as np
import cv2

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

def ckpt_split(file_path,layer_name):
  reader = pywrap_tensorflow.NewCheckpointReader(file_path)
  var_to_shape_map = reader.get_variable_to_shape_map()
  LocalPath = os.getcwd()
  save_path = LocalPath+'/quantity/'
  if not os.path.exists(save_path):
  	os.mkdir(save_path)
  layer_path = save_path+layer_name
  for key in var_to_shape_map:
  	file_list = key.split('/')
  	if('Momentum' in key):
  		continue
  	else:
  		if(layer_name in file_list and 'kernels' in file_list and len(file_list)<=2):
  			weight = reader.get_tensor(key)
  			weight = list(np.ravel(weight))
  			# hex_list,w_ratio = float_to_8bit(weight)
  			weight_file = open(layer_path+'_weight.txt','w+')
  			for i in range(len(weight)):
  				weight_file.write(str(weight[i])+'\n')
  			weight_file.close()
  		elif(layer_name in file_list and 'biases' in file_list):
   	 		bias = reader.get_tensor(key)
   	 		bias = list(np.ravel(bias))
   	 		# hex_list,w_ratio = float_to_8bit(bias)
   	 		bias_file = open(layer_path+'_bias.txt','w+')
   	 		for i in range(len(bias)):
   	 			bias_file.write(str(bias[i])+'\n')
   	 		bias_file.close()
  		elif(layer_name in file_list and 'fold' in key):
   	 		quant = reader.get_tensor(key)
   	 		quant = list(np.ravel(quant))
   	 		# hex_list,q_ratio = float_to_8bit(quant)
   	 		txt_name = layer_path+'_'+file_list[1]+'_'+file_list[2]+'.txt'
   	 		quant_file = open(txt_name,'w+')
   	 		for i in range(len(quant)):
   	 			quant_file.write(str(quant[i])+'\n')
   	 		quant_file.close()

im_path = '/data2/dac_dataset/quantization_tutu/testing/image_2/boat1_000459.jpg'
img = cv2.imread(im_path)
img = img.astype(np.float32, copy=False)
mean = np.array([[[103.939, 116.779, 123.68]]])
img -= mean
img = np.where(img<-127,-127,img)
img = np.where(img>127,127,img)
img = list(np.ravel(img))
hex_list,im_ratio = float_to_8bit(img)
LocalPath = os.getcwd()
save_path = LocalPath+'/quantity/'
if not os.path.exists(save_path):
  os.mkdir(save_path)
im_file = open(save_path+'input.txt','w+')
for i in range(len(hex_list)):
  im_file.write(str(hex_list[i])+'\n')
im_file.close()

print(type(img))
checkpoint_path = "train_original/model.ckpt-0"
layer_list = ['conv1', 'stage_1_1']
for x in layer_list:
  ckpt_split(checkpoint_path,x)


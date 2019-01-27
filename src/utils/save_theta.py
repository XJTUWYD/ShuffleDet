import numpy as np
def savetheta(file_name,variable):
	#np.savetxt('data/'+file_name,variable)
	np.set_printoptions(threshold=np.nan)
	file=open('data/'+file_name,'w')
	file.write(str(variable));
	file.close()
	print(file_name+"has been saved")

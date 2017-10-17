import tifffile as tiff
import numpy as np



def loadbatchfiles(filelist,indexlist,dataMean,dataStd):
		
	inputchanel = 7
	rowsize = 16
	colsize = 16
	
	print dataMean
	print dataStd

	statusPrint = False
	NumData = len(indexlist)
	if NumData > 10000:
		statusPrint = True
		print('Number of data to be loaded %d' % NumData)
		
	x_data = np.float32(np.zeros([len(indexlist),inputchanel,rowsize,colsize]))

	for item in range(NumData):
		arr = tiff.imread(filelist[indexlist[item]])
		for channel in range(inputchanel):
			x_data[item,channel] = np.float32( np.float32(arr[channel]) - np.float32(dataMean[channel] ) / np.float32(dataStd[channel]))
		if statusPrint:
			if item%10000 == 0:
				print('%d of %d data loading complete' % (item,NumData))
	
	return x_data
		
		

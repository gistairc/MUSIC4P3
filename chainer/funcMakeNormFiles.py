# -*- coding: utf-8 -*-
#bin/python
import tifffile as tiff
import numpy as np

import sys
import argparse

import funcReadLocations as readtxt

import sys

parser = argparse.ArgumentParser(description='MakeNormariztionFiles(mean std)')
parser.add_argument('--filelocationspath', default='resource')
parser.add_argument('--positivefilename', default='locationPositiveTrainData.txt')
parser.add_argument('--negativefilename', default='locationNegativeTrainData.txt')
parser.add_argument('--meanstdfilepath', default='meanstd')
args = parser.parse_args()

def mathdatameanstd(pfile,pindex,nfile,nindex,inputchanel):
	
	
	image=[]
	for channel in range(inputchanel):
		band=[]
		image.append(band)
	pNum= len(pindex)
	nNum= len(nindex)
	for NumData in pNum,nNum:
		if NumData is pNum:
			filelist = pfile
			index = pindex
		elif NumData is nNum:
			filelist = nfile
			index = nindex
		for item in range(NumData):
			arr = tiff.imread(filelist[index[item]])
			for channel in range(inputchanel):
				image[channel].append(np.float32(arr[channel]))
			
	mean=[]
	std=[]
	for channel in range(inputchanel):
		data = np.array(image[channel])
		mean.append(np.mean(data))
		std.append(np.std(data))

	return mean,std
	
if __name__ == "__main__":
	
	inputchanel=7
	
	pfilename = args.filelocationspath + "/"+args.positivefilename
	trainPosLocations = readtxt.ReadLocations(pfilename);
	nfilename = args.filelocationspath + "/"+args.negativefilename
	trainNegLocations = readtxt.ReadLocations(nfilename);
	
	negdatasize = len(trainNegLocations)
	posdatasize = len(trainPosLocations)
	
	mean,std = mathdatameanstd(trainPosLocations,range(posdatasize),trainNegLocations,range(negdatasize),inputchanel)
	
	file1 = open(str(args.meanstdfilepath)+".csv",'a')

	for channel in range(inputchanel):
		file1.write(str(mean[channel])+","+str(std[channel])+'\n')
	file1.close()
	
	

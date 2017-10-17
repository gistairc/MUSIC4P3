# -*- coding: utf-8 -*-
#bin/python

import numpy as np
import chainer
from chainer import Function, gradient_check, Variable, optimizers, utils
from chainer import Link, Chain, ChainList
from chainer import serializers
from chainer import cuda
from chainer import computational_graph


import funcReadLocations as readtxt
import megasolar_model_ishii as cnnmodel
import funcLoadBatcFiles as bl

import chainer.functions as F
import chainer.links as L

import os
import csv
import argparse



parser = argparse.ArgumentParser(description='train')
parser.add_argument('--filelocationspath', default='resource')
parser.add_argument('--trainDataPath', default='train')
parser.add_argument('--positivefilename', default='locationPositiveTrainData.txt')
parser.add_argument('--negativefilename', default='locationNegativeTrainData.txt')
parser.add_argument('--meanstdfilepath', default='meanstd')
parser.add_argument('--modelFileNamePath', default='')
args = parser.parse_args()

def get_coordinate_from_csv(csv_file):
    print csv_file
    f = open(str(csv_file),'r')
    file_list = csv.reader(f)
    cor_list = []
    mean=[]
    std=[]
    for row in file_list:
        cor_list.append(row)
    f.close()
    for i in range(len(cor_list)):
		mean.append(np.float32(cor_list[i][0]))
		std.append(np.float32(cor_list[i][1]))   
    return mean,std

def trainMegaSolarCNN():
	
	

	# paths in the local machine
	filelocationspath = args.filelocationspath 

	
	filename = filelocationspath +"/"+args.positivefilename
	trainPosLocations = readtxt.ReadLocations(filename);
	filename = filelocationspath +"/"+args.negativefilename
	trainNegLocations = readtxt.ReadLocations(filename);
	
	
	model = cnnmodel.MegaSolarBN()
	optimizer = optimizers.MomentumSGD(lr=0.001, momentum=0.9)
	optimizer.setup(model)
	model.to_gpu()
	
	negbatchsize = 120
	posbatchsize = 8
	negdatasize = len(trainNegLocations)
	posdatasize = len(trainPosLocations)
	datamean=[]
	datastd=[]
	datamean,datastd=get_coordinate_from_csv(args.meanstdfilepath)
	
	x_train_all_positive = bl.loadbatchfiles(trainPosLocations,range(posdatasize),datamean,datastd)
	x_train_all_negative = bl.loadbatchfiles(trainNegLocations,range(negdatasize),datamean,datastd)
	
	
	NumNegTrainData = 72720
	
	print('Number of Negative samples %d' % (negdatasize))
	print('Number of Positive samples %d' % (posdatasize))
	print('----------------------------------------------------------')
	print('Number of Negative samples at each epoch %d' % (NumNegTrainData))
	print('Number of Positive samples at each epoch %d' % (posdatasize))
	print('----------------------------------------------------------')
	
	batchlooplen = len(range(0, NumNegTrainData, negbatchsize));
	datasize = batchlooplen*(posbatchsize+negbatchsize)
	NUMEPOCH = 2000 
	for epoch in range(NUMEPOCH):
		print('epoch %d' % epoch)
		sum_loss = 0
		sum_accuracy = 0
		mean_loss = np.zeros(NUMEPOCH)
		mean_accuracy = np.zeros(NUMEPOCH)
		negindexes = np.random.permutation(negdatasize)
		posindexes = np.random.permutation(posdatasize)
		posindexes = np.tile(posindexes,(10*8))
		for i in range(0, NumNegTrainData, negbatchsize):

			
			x_train_positive = x_train_all_positive[posindexes[(i/negbatchsize)*posbatchsize:(i/negbatchsize)*posbatchsize+posbatchsize]]
			y_train_positive = np.int32(np.ones(posbatchsize))
			
			x_train_negative = x_train_all_negative[negindexes[i:i+negbatchsize]]
			y_train_negative = np.int32(np.zeros(negbatchsize))
			
			x_train = np.append(x_train_positive,x_train_negative,axis=0)
			y_train = np.append(y_train_positive,y_train_negative)
			
			indexes = np.random.permutation(posbatchsize+negbatchsize)
			x = Variable(cuda.to_gpu(x_train[indexes]))
			t = Variable(cuda.to_gpu(y_train[indexes]))
			

			optimizer.update(model,x,t)
			loss = model(x, t)
								
			sum_loss += loss.data * (posbatchsize+negbatchsize)
			sum_accuracy += model.accuracy.data * (posbatchsize+negbatchsize)
		
		print "EPOCH MEAN TRAIN LOSS and ACCURACY VALUES: "
		mean_loss[epoch] = sum_loss / datasize
		mean_accuracy[epoch] = sum_accuracy / datasize
		print mean_loss[epoch]
		print mean_accuracy[epoch]

		modelFileName = str(args.modelFileNamePath)+str(epoch)+"_iteration.model"
		stateFileName = str(args.modelFileNamePath)+str(epoch)+"_iteration.state"
		serializers.save_npz(modelFileName, model)
		serializers.save_npz(stateFileName, optimizer)
		
	return model,optimizer,loss,mean_loss,mean_accuracy
	
if __name__ == "__main__":
	model,optimizer,loss,mean_loss,mean_accuracy=trainMegaSolarCNN()

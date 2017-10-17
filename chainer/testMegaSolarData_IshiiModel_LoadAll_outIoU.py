# -*- coding: utf-8 -*-
#bin/python

import numpy as np
import chainer
from chainer import Function, gradient_check, Variable, optimizers, utils
from chainer import Link, Chain, ChainList
from chainer import serializers
from chainer import cuda
from chainer import computational_graph
import libtiff

import funcReadLocations as readtxt
import megasolar_model_ishii as cnnmodel
import forwardFunc_IshiiModel as cnnForward
import funcLoadBatcFiles as bl


import chainer.functions as F
import chainer.links as L

import os
import csv
import argparse


parser = argparse.ArgumentParser(description='train')
parser.add_argument('--filelocationspath', default='resource')
parser.add_argument('--testDataPath', default='test')
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

def testMegaSolarCNN():

	

	# paths in the local machine
	filelocationspath = args.filelocationspath 
	
	filename = filelocationspath +"/"+args.positivefilename
	testPosLocations = readtxt.ReadLocations(filename);
	filename = filelocationspath +"/"+args.negativefilename
	testNegLocations = readtxt.ReadLocations(filename);
				

	negbatchsize = 500	
	posdatasize = len(testPosLocations)
	
	negdatasize_part1 = len(testNegLocations) - len(testNegLocations)%negbatchsize
	negdatasize = len(testNegLocations)
	
	
	datamean=[]
	datastd=[]
	datamean,datastd=get_coordinate_from_csv(args.meanstdfilepath)
	
	
	x_test_all_positive = bl.loadbatchfiles(testPosLocations,range(posdatasize),datamean,datastd)
	x_test_all_negative = bl.loadbatchfiles(testNegLocations,range(negdatasize),datamean,datastd)
	
	print('Number of Negative samples %d' % (negdatasize))
	print('Number of Positive samples %d' % (posdatasize))
	print('----------------------------------------------------------')
	

	model = cnnmodel.MegaSolarBN()
	
	MAX_EPOCH = 5000
	accuracy_pos = []
	accuracy_neg = []
	
	iou_all = []
	
	my_pos_thresh = 0.006

	for epoch in range(0,MAX_EPOCH, 1):
		print('EPOCH  %d' % epoch)
		modelFileNamePath=str(args.modelFileNamePath)
		modelFileName = modelFileNamePath+str(epoch)+"_iteration.model"
		serializers.load_npz(modelFileName, model)
		model.to_gpu()
		
		model.train = False
		print "___________________________________________"
		print "Epoch MEAN TEST ACCURACY Positive Samples: "
		x_test_positive = x_test_all_positive
		y_test_positive = np.int32(np.ones(len(testPosLocations)))
		x = Variable(cuda.to_gpu(x_test_positive))
		t = Variable(cuda.to_gpu(y_test_positive))
		respos = cnnForward.forward(model,x)
		cnn_prediction = np.float32(cuda.to_cpu(respos.data[:,1]) > my_pos_thresh)
		cnn_acc =  cnn_prediction.sum()/posdatasize
		accuracy_pos = np.append(accuracy_pos,cnn_acc) 
		print cnn_acc
		x_pos_test = x
		t_pos_test = t
		print "___________________________________________"
		print "Epoch MEAN TEST ACCURACY Negative Samples: "
		
		sum_accuracy = 0		
		for ti in range(0,negdatasize_part1, negbatchsize):

			x_test_negative = x_test_all_negative[ti:ti+negbatchsize]
			y_test_negative = np.int32(np.zeros(negbatchsize))
			x = Variable(cuda.to_gpu(x_test_negative))
			t = Variable(cuda.to_gpu(y_test_negative))
			resneg = cnnForward.forward(model,x)
			cnn_prediction = np.float32(cuda.to_cpu(resneg.data[:,1]) <= my_pos_thresh)
			cnn_acc =  cnn_prediction.sum()/negbatchsize
			sum_accuracy += cnn_acc * (negbatchsize)
		

		x_test_negative = x_test_all_negative[negdatasize_part1:negdatasize]
		y_test_negative = np.int32(np.zeros(len(range(negdatasize_part1,negdatasize))))
		x = Variable(cuda.to_gpu(x_test_negative))
		t = Variable(cuda.to_gpu(y_test_negative))
		resneg = cnnForward.forward(model,x)		
		cnn_prediction = np.float32(cuda.to_cpu(resneg.data[:,1]) <= my_pos_thresh)
		cnn_acc =  cnn_prediction.sum()/len(y_test_negative)
		sum_accuracy += cnn_acc * len(y_test_negative)		
		test_negative_accuracy = sum_accuracy / negdatasize
		
		accuracy_neg = np.append(accuracy_neg,test_negative_accuracy)
		
		print test_negative_accuracy
		x_neg_test = x
		t_neg_test = t	
		TP = np.round(accuracy_pos[-1]*posdatasize)
		FN = posdatasize - TP
		FP = np.round(negdatasize - accuracy_neg[-1]*negdatasize)
		TN = negdatasize - FP
		
		iou_all = np.append(iou_all,TP/(TP+FN+FP))
		print "___________________________________________"
		print "Epoch MEAN TEST Intersection over Union:   "
		print iou_all[-1]
		print "TP:"+str(TP)+"   FN:"+str(FN)
		print "FP:"+str(FP)+"   TN:"+str(TN)
		
	return model, iou_all, x_pos_test, t_pos_test, x_neg_test, t_neg_test 
	
if __name__ == "__main__":
	model, iou_all, x_pos_test, t_pos_test, x_neg_test, t_neg_test=testMegaSolarCNN()

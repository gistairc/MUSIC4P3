import numpy as np
import chainer
from chainer import Function, gradient_check, Variable, optimizers, utils
from chainer import Link, Chain, ChainList

import chainer.functions as F
import chainer.links as L

class MegaSolarBN(chainer.Chain):
    insize = 16
    def __init__(self):
		super(MegaSolarBN, self).__init__(
		    conv1=L.Convolution2D(7,  32, 3, stride=1,pad=0),
		    bn1=L.BatchNormalization(32),
		    conv2=L.Convolution2D(32, 32, 3, stride=1,pad=0),
		    bn2=L.BatchNormalization(32),
		    conv3=L.Convolution2D(32, 32, 3, stride=1,pad=0),
		    bn3=L.BatchNormalization(32),
		    fcl=L.Linear(32*10*10, 2),
		)
		self.train = True

    def __call__(self, x,t):
		h = self.conv1(x)
		h = F.relu(h)
		h = self.bn1(h, test=not self.train)
		
		h = self.conv2(h)
		h = F.relu(h)
		h = self.bn2(h, test=not self.train)
		
		h = self.conv3(h)
		h = F.relu(h)
		h = self.bn3(h, test=not self.train)
		
		h = F.dropout(h, train=self.train)
		
		h = self.fcl(h)
		

		self.pred = F.softmax(h)		
		self.loss = F.softmax_cross_entropy(h, t)
		self.accuracy = F.accuracy(self.pred, t)
		if self.train:
			return self.loss
		else:
			return self.pred
		


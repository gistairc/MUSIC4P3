import numpy as np
import chainer
from chainer import Function, gradient_check, Variable, optimizers, utils
from chainer import Link, Chain, ChainList

import chainer.functions as F
import chainer.links as L

def forward(model,x):
	
	h = model.conv1(x)
	h = F.relu(h)
	h = model.bn1(h, test=True)
	
	h = model.conv2(h)
	h = F.relu(h)
	h = model.bn2(h, test=True)
	
	h = model.conv3(h)
	h = F.relu(h)
	h = model.bn2(h, test=True)
	
	h = model.fcl(h)
	y = F.softmax(h)
		
	model.pred = y
	
	return y

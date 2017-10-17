#!/bin/sh

NORM="mean-std"

python testMegaSolarData_IshiiModel_LoadAll_outIoU.py \
	--filelocationspath "./resource" \
	--positivefilename "locationPositiveTestData.txt" \
	--negativefilename "locationNegativeTestData.txt" \
	--meanstdfilepath ${NORM}".csv" \
	--modelFileNamePath "./result/newmegasolarCNN_NS_72720_"

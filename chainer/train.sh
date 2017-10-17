#!/bin/sh

NORM="mean-std"

mkdir result

python funcMakeNormFiles.py \
	--filelocationspath "./resource" \
	--positivefilename "locationPositiveTrainData.txt" \
	--negativefilename "locationNegativeTrainData.txt" \
	--meanstdfilepath ${NORM}

python trainMegaSolarData_IshiiModel_LoadAll.py \
	--filelocationspath "./resource" \
	--positivefilename "locationPositiveTrainData.txt" \
	--negativefilename "locationNegativeTrainData.txt" \
	--meanstdfilepath ${NORM}".csv"  \
    --modelFileNamePath "./result/newmegasolarCNN_NS_72720_"

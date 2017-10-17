#!/bin/sh

GPU=1

MODEL=ishiinet
echo ${MODEL}

for i in 15
do
    th train-cnn.lua \
        -save ${MODEL}_p1n${i} \
        -gpu ${GPU} \
        -trainDataPath ./resource/train \
        -testDataPath ./resource/val \
        -batchNorm \
        -dropout \
        -batchSize 128\
        -batchSizeNM 32 \
        -maxEpoch 3000 \
        -negativeRatio ${i} \
        -saveInterval 10 \
        -optimization SGD \
        -trainnorm ${MODEL}_meanstd.csv \
        | tee log_${MODEL}_p1n${i}.txt
done

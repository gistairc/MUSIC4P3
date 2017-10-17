#!/bin/sh

LIST=test_hdf5_list.txt


DATA_PATH='./resource'
MODE=ishiinet

NETWORK_PATH=./${MODEL}_p1n15
NETWORK='cnn_nm_ep2000.net'
OUTPUT_NETWORK='cnn_nm_ep2000_cpu.net'

th convertGPUToCPU.lua \
    -network ${NETWORK_PATH}/${NETWORK} \
    -outputNetwork ${NETWORK_PATH}/${OUTPUT_NETWORK} \
    -gpu 1

for THRESHOLD in 0.999
do
    cat ${LIST} | while read HDF5
    do
        th test-cnn-cpu.lua \
            -testDataPath ${DATA_PATH}/test/${HDF5} \
            -testBatchSize 4000 \
            -network ${NETWORK_PATH}/${OUTPUT_NETWORK} \
            -threshold ${THRESHOLD} \
            -meanstd_file ${MODEL}_meanstd.csv \
            | tee ${NETWORK_PATH}/log_th${THRESHOLD}_${HDF5}.txt
    done

    for i in ${NETWORK_PATH}/log_th${THRESHOLD}_*${MODE}.hdf5.txt
    do
        echo ${i}
        cat ${i} >> ${NETWORK_PATH}/${MODE}_th${THRESHOLD}_all.txt
    done

    python ComputeCM.py \
        --log_file ${NETWORK_PATH}/${MODE}_th${THRESHOLD}_all.txt \
        > ${NETWORK_PATH}/${MODE}_th${THRESHOLD}_all_cm.txt
    
done




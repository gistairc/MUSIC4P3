
# coding: utf-8

import string
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--log_file', default='')
args = parser.parse_args()


f = open(args.log_file, 'r')
str_list = []
for line in f:
    str_list.append(line)
f.close()


def get_cm_value(str_line):
    tmp_str = []
    for i in range(len(str_line)):
        if str_line[i] != '':
            tmp_str.append(str_line[i])
 
    return int(tmp_str[1]), int(string.split(tmp_str[2], ']')[0])

result_list = []
for i in range(len(str_list)):
    tmp_line = string.split(str_list[i],' ')
    if ('test' in tmp_line) and ('all' in tmp_line):
        result_list.append([i,
                            get_cm_value(string.split(str_list[i+2])),
                            get_cm_value(string.split(str_list[i+3]))])

confusion = [[0,0],[0,0]]

IOU_list = []
for i in range(len(result_list)):
    tp = result_list[i][1][0]
    fn = result_list[i][1][1]
    fp = result_list[i][2][0]
    tn = result_list[i][2][1]
    for j in [0,1]:
        for k in [0,1]:
            confusion[j][k] = confusion[j][k] + result_list[i][j+1][k]   

iou = float(confusion[0][0])/float(confusion[0][0]+confusion[0][1]+confusion[1][0])
recall = float(confusion[0][0])/float(confusion[0][0]+confusion[0][1])
precision = float(confusion[0][0])/float(confusion[0][0]+confusion[1][0])


print 'Confusion Matrix:'
print confusion

print 'IOU:'+str(iou)
print 'Precision:'+str(precision)
print 'Recall:'+str(recall)


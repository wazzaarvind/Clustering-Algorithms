#!/usr/bin/env python
import numpy as np
from operator import itemgetter
import sys

outputMatrix = []
flag = 0
dictCluster = dict()
sumForCentroid = []
newCluster = -2
lenMatrix = -1
# input comes from STDIN
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()

    # parse the input we got from mapper.py
    cluster, index, matrixString = line.split('\t')

    matrix = [float(x) for x in matrixString.split(',')]
    matrix = np.array(matrix)
    lenMatrix = len(matrix)

    cluster = int(cluster)
    index = int(index)

    if(newCluster!=cluster):
        if(newCluster!=-2):
            newCentroid = sumForCentroid/len(outputMatrix)
            print ('%s\t%s\t%s' % (newCluster, ','.join(map(str,outputMatrix)),','.join(map(str,newCentroid))))
        outputMatrix = []
        sumForCentroid = np.zeros(len(matrix))
        newCluster = cluster

    outputMatrix.append(index)

    sumForCentroid += matrix


    
newCentroid = sumForCentroid/len(outputMatrix)
print ('%s\t%s\t%s' % (newCluster, ','.join(map(str,outputMatrix)),','.join(map(str,newCentroid))))



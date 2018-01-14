#!/usr/bin/env python
import numpy as np
import sys
import os
import math


#numberClusters = 5
numberClusters = os.environ['numberClusters']

unformattedMatrix=np.loadtxt(sys.stdin)
centroidList = unformattedMatrix[:int(numberClusters),:]


matrix = unformattedMatrix[int(numberClusters):,:]



i = 0
cluster = np.zeros(matrix.shape[0],dtype='int')
squaredDistance = 0


while(i < matrix.shape[0]):
    k = 0
    maxCurrentDistance = 10000

    while(k < len(centroidList)):
        j = 0;
        while(j < matrix.shape[1]):
            squaredDistance += (matrix[i][j] - centroidList[k][j])**2
            j += 1
        distance = math.sqrt(squaredDistance)
        if(maxCurrentDistance > distance):
            maxCurrentDistance = distance
            cluster[i] = int(k)
        squaredDistance = 0
        k += 1
    i += 1
    
i = 0
while(i<len(cluster)):
    print ('%s\t%s\t%s' % (cluster[i], i, ','.join(map(str, matrix[i]))))
    i += 1




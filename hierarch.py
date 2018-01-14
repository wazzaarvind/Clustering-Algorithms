from __future__ import division
import numpy as np
import math
import random
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from random import randint
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
import copy



def genHCluster(distanceValues):
	newIndices = np.unravel_index(distanceValues.argmin(), distanceValues.shape)
	if(distanceValues[newIndices[0]][newIndices[1]]==10000):
		return []

	distanceValues[newIndices[0]][newIndices[1]] = 10000
	return newIndices


def mergeCluster(matrix,distanceValues,newIndices):
	
	j = 0
	while(j<matrix.shape[0]):

		if(distanceValues[newIndices[0]][j] > distanceValues[newIndices[1]][j]):
			if(newIndices[0]!=j):
				distanceValues[newIndices[0]][j] = distanceValues[newIndices[1]][j]
				distanceValues[j][newIndices[0]] = distanceValues[newIndices[1]][j]
		distanceValues[newIndices[1]][j] = 10000
		distanceValues[j][newIndices[1]] = 10000
		j += 1


def readFile(fileName):
	global clusterLimit
	final_cluster=[]
	file=open(fileName,'r')
	line=file.readline()
	distanceValues = []

	columnsLimit=len(line.strip().split("\t"))
	clusters=np.loadtxt(fileName,usecols=range(1,2)) # To obtain total number of clusters

	x = np.amax(clusters) #Getting the total number of clusters

	matrix=np.loadtxt(fileName,usecols=range(2,columnsLimit)) #Load attributeds to np matrix

	i = 0
	index = -1
	maxDistance = 0;


	while(i < matrix.shape[0]):
		j = 0
		k = 0
		distanceValues.append([])
		while(k < matrix.shape[0]):
			squaredDistance = 0
			while(j < matrix.shape[1]):
				squaredDistance += (matrix[i][j] - matrix[k][j])**2
				j += 1
			j = 0
			distance = math.sqrt(squaredDistance)
			distanceValues[i].append(distance)
			k += 1
		i += 1
	distanceValues = np.array(distanceValues)

	newkey = 0

	i = 0
	while(i < matrix.shape[0]):
		distanceValues[i][i] = 10000
		i += 1

	clusterCount = 1
	clusterDict = dict()

	for i in range(0,matrix.shape[0]):
		clusterDict[clusterCount] = []
		clusterDict[clusterCount].append(i)
		clusterCount += 1
	while True:

		newIndices = genHCluster(distanceValues)


		if(newIndices == []):
			break

		flag = 0
		flag1 = 0
		flag2 = -1
		tempUseList = []
		for i in list(clusterDict):
			if newIndices[0] in clusterDict[i]:
				tempUseList = list(clusterDict[i])
				del(clusterDict[i])
				tempUseList.append(newIndices[1])
				set1 = set(tempUseList)
				flag = 1
			elif newIndices[1] in clusterDict[i]:
				tempUseList = list(clusterDict[i])
				del(clusterDict[i])
				tempUseList.append(newIndices[0])
				set2 = set(tempUseList)
				flag1 = 2
		

		if(flag == 1 and flag1 == 2):
			unionSet = set1 | set2
			clusterDict[clusterCount] = list(unionSet)
			clusterCount += 1
			flag = 0
			flag1 = 0
			flag2 = 1

		elif(flag == 1):
			clusterDict[clusterCount] = list(set1)
			clusterCount += 1
			flag = 0
			flag2 = 1

		elif(flag1 == 2):
			clusterDict[clusterCount] = list(set2)
			clusterCount += 1
			flag1 = 0
			flag2 = 1

		
		if(len(clusterDict) == clusterLimit):
			print('The final Clusters are')
			for key,value in clusterDict.items():
				updatedValue = []
				for singleGene in value:
					updatedValue.append(singleGene+1)
				final_cluster.append(updatedValue)
				
				
			for y in range(0,len(final_cluster)):
				print("Cluster %d" %(y+1) , final_cluster[y])	
				
				
			break
			
			

		mergeCluster(matrix,distanceValues,newIndices)
	

		

		
	groundJaccard = np.zeros(shape=(matrix.shape[0],matrix.shape[0]))
	i = 0
	while(i<matrix.shape[0]):
		j = 0
		while(j<matrix.shape[0]):
			if(i == j):
				groundJaccard[i][j] = 1
			elif(clusters[i]==clusters[j]):# and clusters[i]!=-1):
				groundJaccard[i][j] = 1
			else:
				groundJaccard[i][j] = 0
			j += 1
		i += 1


	hierarchJaccard = np.zeros(shape=(matrix.shape[0],matrix.shape[0]))
	i = 0
	ikey = -1
	jkey = -5
	while(i<matrix.shape[0]):
		for key in clusterDict:
			if i in clusterDict[key]:
				ikey = key
				break
		j = 0
		while(j<matrix.shape[0]):
			for key in clusterDict:
				if j in clusterDict[key]:
					jkey = key
					break
			if ikey == jkey:
				hierarchJaccard[i][j] = 1
			else:
				hierarchJaccard[i][j] = 0
			jkey = -5
			j += 1
		ikey = -1
		i += 1

	agree = 0
	disagree = 0 
	i = 0
	while(i<matrix.shape[0]):
		j = 0
		while(j<matrix.shape[0]):
			if(groundJaccard[i][j] == 1 and (groundJaccard[i][j] == hierarchJaccard[i][j])):
				agree += 1
			if(not((groundJaccard[i][j] == 0) and (hierarchJaccard[i][j]==0))):
				disagree += 1
			j += 1
		i += 1

	jaccard = agree/disagree
	print("Jaccard for Hierarchial Clustering : " , jaccard)


	pca = PCA(n_components=2).fit_transform(matrix)

	x2 = pca[:,0]
	y2 = pca[:,1]

	colors = cm.Set1(np.linspace(0, 1, len(clusterDict)))
	plt.figure()
	ax2=plt.subplot(1,1,1)
	count=1
	list_of_ids=[]
	for key,value in clusterDict.items():
		list_of_ids=value

		p1=[]
		p2=[]
		for l in range(0,len(list_of_ids)):
			p1.append(x2[list_of_ids[l]])
			p2.append(y2[list_of_ids[l]])

		ax2.scatter(p1,p2,s=50,color=colors[count-1] , label=count )
		count+=1
	
	ax2.legend(scatterpoints=1 , loc='best')
	titleString = "Hierarchial clustering algorithm for "+inputFileName+" with no. of clusters = "+str(clusterLimit)
	plt.title(titleString)
	plt.xlabel("Principal Component 1")
	plt.ylabel("Principal Component 2")
	plt.show()
	
	
# Take the input for filename and number of clusters	
inputFileName = input("Enter the file name for input: ")
clusterLimit = int(input("Enter the number of clusters: "))
readFile(inputFileName)
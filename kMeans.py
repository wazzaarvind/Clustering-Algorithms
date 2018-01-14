import numpy as np
import math
import random
from sklearn.metrics.cluster import adjusted_rand_score
from random import randint
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA



def repeatCluster(matrix,centroidList):
	i = 0
	cluster = np.zeros(matrix.shape[0])
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
				cluster[i] = k
			squaredDistance = 0
			k += 1
		i += 1
	return cluster

def repeatFindCentroid(cluster,matrix,centroidList):

	tempNewCentroidList = np.zeros(shape=(centroidList.shape[0],centroidList.shape[1]))
	specificClusterIndices = []
	count = 0
	while(count < len(centroidList)):
		specificClusterIndices.append([])
		specificClusterIndices[count] = np.where(cluster == count)[0]
		count += 1
	specificClusterIndices = np.array(specificClusterIndices)

	i = 0

	while(i < len(specificClusterIndices)):
		j = 0
		while(j < matrix.shape[1]):
			k = 0
			sumforCentroid = 0
			while(k < len(specificClusterIndices[i])):
				sumforCentroid += matrix[specificClusterIndices[i][k]][j]
				k += 1
			tempNewCentroidList[i][j] = sumforCentroid/len(specificClusterIndices[i])
			sumforCentroid = 0
			j += 1
		i += 1

	return tempNewCentroidList



def readFile(fileName):
	file=open(fileName,'r')
	line=file.readline()
	distanceValues = []

	columnsLimit=len(line.strip().split("\t"))
	clusters=np.loadtxt(fileName,usecols=range(1,2)) # To obtain total number of clusters

	#x = np.amax(clusters) #Getting the total number of clusters
	x=len(inputCentroidsList)

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
	

	centroidList = []
	count = 0
	randomIndices = []
	'''while(count<x):
		randomInt = np.random.randint(0, matrix.shape[0])
		if(randomInt not in randomIndices):
			randomIndices.append(randomInt)
			centroidList.append(matrix[randomInt])
			count += 1'''
	for centroidValue in inputCentroidsList:
		centroidList.append(matrix[centroidValue])

	#centroidList.append(matrix[3])
	#centroidList.append(matrix[4])
	#centroidList.append(matrix[8])
	#centroidList.append(matrix[35])
	#centroidList.append(matrix[237])

	centroidList = np.array(centroidList)



	# Now do the actual K Means Algorithm

	
	newCount = 0
	while True:
		newCount += 1
		cluster = repeatCluster(matrix,centroidList)

		newCentroidList = repeatFindCentroid(cluster,matrix,centroidList)

		if(np.array_equal(centroidList, newCentroidList) or newCount == iterationLimit):
			print("Final Centroids :",centroidList)
			break
		else:
			centroidList = newCentroidList


	cluster = cluster + 1
	
	#print(cluster)
	
	members=[]
	for i in range(0,int(x)):
		members=[]
		members=np.where(cluster==i+1)[0]
		members=members+1
		
		print("Cluster %d" %(i+1) ,members )


	colors = cm.Set1(np.linspace(0, 1, x))

	pca = PCA(n_components=2).fit_transform(matrix)

	x2 = pca[:,0]
	y2 = pca[:,1]

	plt.figure()
	ax2=plt.subplot(1,1,1)
	for l in range(1,int(x+1)):    # looping through class labels
		p1=[]	
		p2=[]
		for nums in range(0,len(x2)):	# looping through points x 
			if cluster[nums]==l:        # if number belongs to the particular class label
			 	p1.append(x2[nums])
			 	p2.append(y2[nums])
		ax2.scatter(p1,p2,s=50,color=colors[l-1] , label=l )
					 	
		
	ax2.legend(scatterpoints=1 , loc='best')
	titleString = "K-Means clustering algorithm for "+inputFileName+" with K = "+str(len(inputCentroidsList))
	plt.title(titleString)
	plt.xlabel("Principal Component 1")
	plt.ylabel("Principal Component 2")
	plt.show()



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


	ourClusterJaccard = np.zeros(shape=(matrix.shape[0],matrix.shape[0]))
	i = 0
	while(i<matrix.shape[0]):
		j = 0
		while(j<matrix.shape[0]):
			if(cluster[i]==cluster[j]):
				ourClusterJaccard[i][j] = 1
			else:
				ourClusterJaccard[i][j] = 0
			j += 1
		i += 1

	agree = 0
	disagree = 0 
	i = 0
	while(i<matrix.shape[0]):
		j = 0
		while(j<matrix.shape[0]):
			if(groundJaccard[i][j] == ourClusterJaccard[i][j]):
				agree += 1
			j += 1
			disagree += 1
		i += 1

	randIndex = agree/disagree


	agree = 0
	disagree = 0 
	i = 0
	while(i<matrix.shape[0]):
		j = 0
		while(j<matrix.shape[0]):
			if(groundJaccard[i][j] == 1 and (groundJaccard[i][j] == ourClusterJaccard[i][j])):
				agree += 1
			if(not((groundJaccard[i][j] == 0) and (ourClusterJaccard[i][j]==0))):
				disagree += 1
			j += 1
		i += 1

	jaccard = agree/disagree
	print("\n\nJaccard Value for KMeans is : ",jaccard)

inputFileName = input("Enter the file name for input: ")
inputCentroidsList = []
zeroInputCentroidList = []
zeroInputCentroidList = input("Enter the list of input centroids: ").split(",")
zeroInputCentroidList = list(map(int, zeroInputCentroidList ))
for singleCentroid in zeroInputCentroidList:
	inputCentroidsList.append(singleCentroid-1)
iterationLimit = input("Iteration limit: ")
readFile(inputFileName)
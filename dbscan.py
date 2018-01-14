from __future__ import division
import numpy as np
import math
import random
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from random import randint
import matplotlib.pyplot as plt
import matplotlib .cm as cm
from sklearn.decomposition import PCA


matrix = []
disMatrix = []
clusterMembers = []
visitedSet = set()
noiseSet = set()
clusters = []
def distanceCal(fileName):
	# Opens file cho.txt
	file = open(fileName,'r') 
	global clusters
	clusters = np.loadtxt(fileName,usecols=range(1,2))
	line = file.readline()
	distanceValues = [] # This is to hold the distance of all points from all points as a list of list. Further lists are added 
	columnsLimit = len(line.strip().split("\t")) # Parses single line and returns the number of spaces
	global disMatrix 
	disMatrix = []
	# To obtain total number of clusters in fileName
	clusters = np.loadtxt(fileName,usecols=range(1,2))
	x = np.amax(clusters) # Getting the total number of clusters; np.amax(): Return the maximum of an array or maximum along an axis.
	matrix=np.loadtxt(fileName,usecols=range(2,columnsLimit)) # Load attributes to np matrix
	for i in range(0, len(matrix)):
		disMatrix.append([])
		for j in range(0, len(matrix)):
			dis = np.linalg.norm(matrix[i]-matrix[j])
			disMatrix[i].append(dis)

	disMatrix = np.array(disMatrix)
	return matrix 

def regionQuery(i, eps):
	neighbors = []
	for point in range(0, len(matrix)):
		#if point != i:
		if disMatrix[i][point]<=eps:
			neighbors.append(point)
	return neighbors

def expandCluster(i, neighbors, clusterNo, eps, minpts):
	clusterMembers[clusterNo-1].append(i)
	flag = 0
	iteration = 0
	size=len(neighbors)
	while (iteration < size):
		flag = 0
		if neighbors[iteration] not in visitedSet:
			flag = 0
			visitedSet.add(neighbors[iteration])
			neighborsOfPoint = regionQuery(neighbors[iteration], eps)
			if len(neighborsOfPoint) >= minpts:
				neighbors = neighbors + [i for i in neighborsOfPoint if i not in neighbors]
				size = len(neighbors)
		for cluster in clusterMembers:
			for onePoint in cluster:
				if neighbors[iteration] == onePoint:
					flag = 1
					break
		if flag == 0:
			clusterMembers[clusterNo-1].append(neighbors[iteration]) 
			if(neighbors[iteration] in noiseSet):
				noiseSet.remove(neighbors[iteration])

		iteration += 1
def dbscan(eps, minpts):
	clusterNo = 0
	for i in range(0, len(matrix)):
		if i not in visitedSet:
			visitedSet.add(i)
			neighbors = regionQuery(i, eps)
			if (len(neighbors)) < minpts:
				noiseSet.add(i)
			else:
				clusterNo+=1
				clusterMembers.append([])
				expandCluster(i, neighbors, clusterNo, eps, minpts)

# Need filename from the user
inputFile = input("Enter the file to take input from: ")
matrix = distanceCal(inputFile)
# Need to take input parameters from user
inputEpsilonDistance = float(input("Epsilon distance: "))
inputMinPts = int(input("Minimum number of points: "))
dbscan(inputEpsilonDistance, inputMinPts)
clusterMembers.append(list(noiseSet))

noiseSet = [x+1 for x in noiseSet]

print_clusterMembers=[]
for i in range(0,len(clusterMembers)):
	print_clusterMembers.append([])
	for j in range(0,len(clusterMembers[i])):
		print_clusterMembers[i].append(clusterMembers[i][j]+1)



print("No of clusters: ",len(print_clusterMembers)-1,"\n\n")

print("The Clusters are as follows : ")
for i in range(0,len(print_clusterMembers)-1):
	print("Cluster %d:" %(i+1) , print_clusterMembers[i] )
	

#print("NoiseSet: ",noiseSet , "\n")
#print("NoiseSet size: ",len(noiseSet),"\n")


colors = cm.Set1(np.linspace(0, 1, len(clusterMembers)))


labels=[]
for xyz in range(0,len(clusterMembers)-1):
	labels.append(xyz+1)

labels.append('Noise')
pca = PCA(n_components=2).fit_transform(matrix)

x2 = pca[:,0]
y2 = pca[:,1]

plt.figure()
ax2=plt.subplot(1,1,1)


if(len(noiseSet)==0):
	clusterLen=len(clusterMembers)-1
else:
	clusterLen = len(clusterMembers)

for i in range(0,clusterLen):    # looping through class labels
	p1=[]	
	p2=[]
	for j in range(0,len(clusterMembers[i])):	# looping through points x 
		p1.append(x2[clusterMembers[i][j]])
		p2.append(y2[clusterMembers[i][j]])
	ax2.scatter(p1,p2,s=50,color=colors[i] , label=labels[i])


	
ax2.legend(scatterpoints=1 , loc='best')
titleString = "Density based scan for "+inputFile+" with Epsilon = "+str(inputEpsilonDistance)+" and Min Points = "+str(inputMinPts)
plt.title(titleString)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()




dbJaccard = np.zeros(shape=(matrix.shape[0],matrix.shape[0]))
i = 0
ikey = -1
jkey = -5
ones = 0
while(i<matrix.shape[0]):
	for index in range(0,len(clusterMembers)):
		if i in clusterMembers[index]:
			ikey = index
			break

	j = 0

	while(j<matrix.shape[0]):
		flagj = 0
		for index in range(0,len(clusterMembers)):
			if j in clusterMembers[index]:
				jkey = index
				flagj = 1
				break
		if(i == j):
			dbJaccard[i][j] = 1
		elif ikey == jkey:
			dbJaccard[i][j] = 1
			ones += 1
		else:
			dbJaccard[i][j] = 0
		jkey = -5
		j += 1
	ikey = -1
	i += 1

groundJaccard = np.zeros(shape=(matrix.shape[0],matrix.shape[0]))
i = 0
while(i<matrix.shape[0]):
	j = 0
	while(j<matrix.shape[0]):
		if(i == j):
			groundJaccard[i][j] = 1
		if(clusters[i]==clusters[j]):# and clusters[i]!=-1):
			groundJaccard[i][j] = 1
		else:
			groundJaccard[i][j] = 0
		j += 1
	i += 1

agree = 0
disagree = 0 
i = 0
while(i<matrix.shape[0]):
	j = 0
	while(j<matrix.shape[0]):
		if(groundJaccard[i][j] == 1 and (groundJaccard[i][j] == dbJaccard[i][j])):
			agree += 1
		if(not((groundJaccard[i][j] == 0) and (dbJaccard[i][j]==0))):
			disagree += 1
		j += 1
	i += 1

jaccard = agree/disagree
print("\n\nJaccard for DBScan is :",jaccard)



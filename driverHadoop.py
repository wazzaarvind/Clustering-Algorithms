import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA



def readFile(fileName):
	file=open(fileName,'r')
	line=file.readline()
	distanceValues = []

	columnsLimit=len(line.strip().split("\t"))
	clusters=np.loadtxt(fileName,usecols=range(1,2)) # To obtain total number of clusters

	#x = np.amax(clusters) #Getting the total number of clusters
	x = len(inputCentroidsList)

	matrix=np.loadtxt(fileName,usecols=range(2,columnsLimit))

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

		'''
	centroidList.append(matrix[378])
	centroidList.append(matrix[55])
	centroidList.append(matrix[51])
	centroidList.append(matrix[35])
	centroidList.append(matrix[237])
		'''
	centroidList = np.array(centroidList)


	finalList = np.concatenate((centroidList,matrix))

	outputName = "dataToMapper.txt"
	outputFile = open("dataToMapper.txt",'ab')
	np.savetxt(outputName,finalList)
	x = str(int(x))
	count = 0
	cluster = np.zeros(matrix.shape[0])

	while True:
		os.system("hadoop fs -rm  /dataToMapper.txt")
		os.system("hadoop fs -put dataToMapper.txt /")
		os.system("hadoop fs -rm -r /output_folder")

		####### $IMPORTANT$
		# USER SHOULD CHANGE THIS COMMAND AS PER SYSTEM HADOOP DIRECTORY
		os.system("hadoop jar /usr/local/Cellar/hadoop/2.8.1/libexec/share/hadoop/tools/lib/hadoop-streaming-2.8.1.jar -Dmapreduce.job.maps="+x+" -Dmapreduce.job.reduces="+x+" -files ./mapper.py,./reducer.py -mapper ./mapper.py -reducer ./reducer.py -cmdenv numberClusters=" + x + " -input  /dataToMapper.txt -output /output_folder")
		count += 1

		output = subprocess.check_output("hadoop fs -cat /output_folder/part-*", shell=True)
		output = output.decode('ascii').strip()
		lines = output.split('\n')
		oldCentroidList = np.copy(centroidList)
		centroidList = np.zeros(shape=(int(x),matrix.shape[1]))
		finalList = []
		centroidCount = 0
		for line in lines:
			clusterNo = line.split('\t')[0]
			indices = line.split('\t')[1]


			indicesList = indices.split(',')
			indicesList = np.array(indicesList,dtype='int')
			for everyIndex in indicesList:
				cluster[everyIndex] = clusterNo

			temporaryList = line.split('\t')[2]
			centroidList[int(clusterNo)] = [float(x) for x in temporaryList.split(',')]


		if(np.array_equal(centroidList, oldCentroidList)):
			print("Final Centroids : ")
			print(centroidList)
			break
		finalList = np.concatenate((centroidList,matrix))
		np.savetxt(outputName,finalList)

	groundJaccard = np.zeros(shape=(matrix.shape[0],matrix.shape[0]))
	i = 0
	while(i<matrix.shape[0]):
		j = 0
		while(j<matrix.shape[0]):
			if(i==j):
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
			if(groundJaccard[i][j] == 1 and (groundJaccard[i][j] == ourClusterJaccard[i][j])):
				agree += 1
			if(not((groundJaccard[i][j] == 0) and (ourClusterJaccard[i][j]==0))):
				disagree += 1
			j += 1
		i += 1

	jaccard = agree/disagree
	print("The found Jaccard coefficient is :",jaccard)



	colors = cm.Set1(np.linspace(0, 1, x))

	pca = PCA(n_components=2).fit_transform(matrix)

	x2 = pca[:,0]
	y2 = pca[:,1]
	cluster += 1
	plt.figure()
	ax2=plt.subplot(1,1,1)
	for l in range(1,int(x)+1):    # looping through class labels
		p1=[]	
		p2=[]
		for nums in range(0,len(x2)):	# looping through points x 
			if cluster[nums]==l:        # if number belongs to the particular class label
			 	p1.append(x2[nums])
			 	p2.append(y2[nums])
		ax2.scatter(p1,p2,s=50,color=colors[l-1] , label=l )
					 	
		
	ax2.legend(scatterpoints=1 , loc='best')
	plt.show()
	
inputFile = input("Enter the file name: ")
inputCentroidsList = []
zeroInputCentroidList = []
zeroInputCentroidList = input("Enter the list of input centroids: ").split(",")
zeroInputCentroidList = list(map(int, zeroInputCentroidList ))
for singleCentroid in zeroInputCentroidList:
	inputCentroidsList.append(singleCentroid-1)
readFile(inputFile)



K – Means



1.The program execution is done by running the kMeans.py using Python 3

>python3 kMeans.py


2.The program will ask for the following input parameters to be supplied : 

Enter the file name for input:  Text file to be given as input
Enter the list of input centroids: Initial data points(gene id’s) to be given as centroids separated by commas (Example: 3,5,7) 
Iteration Limit : Specify the number of iterations for which the kMeans algorithm is to be executed. 

3.The centroid values , Jaccard coefficient and the clusters formed after k iterations are displayed.

 
4.The program will also generate the plot showing the clusters obtained after running the program for k iterations. 


----------------------------------------------------------------------------------------------	
			

Hierarchial Agglomerative Clustering



1.The program execution is done by running the hierarch.py using Python 3

>python3 hierarch.py


2.The program will ask for the following input parameters to be supplied : 

Enter the file name for input:  Text file to be given as input
Enter the number of clusters:  Number of clusters required


3.The total number of clusters formed and the Jaccard coefficient will be displayed .


4.The program will also generate the plot showing the data points in their respective clusters. 


------------------------------------------------------------------------------------------------


			
Density Based Clustering



1. The program execution is done by running the dbscan.py using Python 3

>python3 dbscan.py


2. The program will ask for the following input parameters to be supplied : 

Enter the file name to take input from:  Text file to be given as input
Epsilon Distance :   Epsilon distance (eps)
Minimum Number of Points:   Minimum points (minpts)


3.The total number of clusters formed and the Jaccard coefficient will be displayed .


4.The program will also generate the plot showing the data points in their respective clusters and also noise points(outliers) if any.

 
--------------------------------------------------------------------------------------------------




Hadoop MapReduce K-Means Clustering Algorithm



Place the driverHadoop.py, mapper.py, redcuer.py, the input text file in the same folder from which you are to run the code. 

1. Open file driverHadoop.py. Go to line 60 or search for "$IMPORTANT$" and replace #localaddressforhadoopstreamingjarfile# with an appropriate address for the hadoop path as in your system.
hadoop jar #localaddressforhadoopstreamingjarfile# -Dmapreduce.job.maps="+x+" -Dmapreduce.job.reduces="+x+" -files ./mapper.py,./reducer.py -mapper ./mapper.py -reducer ./reducer.py -cmdenv numberClusters=" + x + " -input  /dataToMapper.txt -output /output_folder"

2. Run the Python script diverHadoop.py.
> python3 driverHadoop.py

3. Enter the file name which you are going to use as an input.
> Enter the file name: $fileName$

4. Enter a list of centroids to start with. Make them comma seperated. 
> Enter the list of input centroids: x,y,z,p,q,r

5. If the name of the input text file or the name of the output folder or the path in which the input is taken from or the path where the output is taken to needs to be changed refer to lines 48, 49, 56, 57, 58, 62, 65.















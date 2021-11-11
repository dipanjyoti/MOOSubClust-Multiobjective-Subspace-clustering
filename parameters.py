import random
import math
from datetime import datetime
import time
import numpy as np
from collections import OrderedDict
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import pygmo as pg

start_time = time.clock()
crowding_size=10 # Number of solution size

mydata_list_of_list=[] # total data set
sampleData=[] # a sample of data taken for clustering

#For Real data
NoOfClasses =6
dimension =9   

ExpCluster= 3 * NoOfClasses

#sample_size=ExpCluster * 25
sample_size=150
#SDmax=ExpCluster * dimension
SDmax=80

NoOfIteration=2001
#NoOfIteration= SDmax * ExpCluster  # divided by 10 bcoz at a time no_of_sol is 10 (crowding_size)
#NoOfIteration= 10 * SDmax * ExpCluster 

z_value=80 #loop in z
init_value=20 # numder of timer initial onechild will run

F1_len=0
Iteration=0

membershipList=[] # Sample data belongs to which cluster
membershipListAllData=[]
membershipListAllDataCopy=[]
mean=[] # mean of total data

trueLevel=[]
trueLevel1=[]

Model_dict={}
genotype_dict={}
Model_final_dict={}
genotype_final_dict={}

print
print 'dataset= glass.arff'
print 'Iteration=', NoOfIteration

dataCount=0
readfile=open('glass.arff','r')   #  breast is Real world dataset
for line in readfile.readlines():
	line = line.strip()
	my_list=[]
	for word in line.split(',')[0:len(line.split(','))-1]:
		my_list.append(word)
	mydata_list_of_list.append(my_list)
	dataCount=dataCount+1
	for word in line.split(',')[(len(line.split(','))-1):len(line.split(','))]:
		trueLevel1.append(word)
trueLevel = [int(x) for x in trueLevel1]
#print (trueLevel)

print 'datacount=',dataCount
print
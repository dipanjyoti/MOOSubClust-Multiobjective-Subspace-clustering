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
from parameters import *
 
#function to standardization of dataset

def normalizedData():

	deviatedData_list_of_list=[] # data - mean
	deviation=[]

	for i in range(dataCount): # Initializing
		deviatedData_list_of_list.append([])
		for j in range(dimension):
			deviatedData_list_of_list[i].append(0)

	for d in range(dimension): # Mean data calculation
		add=0.0
		for data in range(dataCount):
			add=add+float(mydata_list_of_list[data][d])
		mean.append(add/(dataCount+0.0))
	#print ('mean=',mean)

	for data in range(dataCount): # deviation calculation of each data
		for d in range(dimension):
			deviatedData_list_of_list[data][d]=np.square(float(mydata_list_of_list[data][d])-float(mean[d]))

	for d in range(dimension): # Mean deviation
		addition=0.0
		for data in range(dataCount):
			addition+=float(deviatedData_list_of_list[data][d])
		deviation.append(np.sqrt(addition/ (dataCount+0.0)))
	#print ('deviation=',deviation)

	for i in range(dataCount): # Normalized data
		for j in range(dimension):
			if float(deviation[j])==0.0:
				mydata_list_of_list[i][j]=0.0
			else:
				mydata_list_of_list[i][j]=(float(mydata_list_of_list[i][j])-float(mean[j]))/(float(deviation[j]+0.0))
	#print ('standard data=',mydata_list_of_list)


# Function to Check Empty Model
def Check_Empty_Model(M,row_num):
	count=0
	Model=M
	for d in range(dimension):
		if Model[row_num][d]!=0:
			break
		else:
			count=count+1
	if count==dimension:
		return True
	else:
		return False

#function to calculate Number of cluster
def Cal_Noof_Cluster():
	countCluster=0
	for m in range(SDmax):

		for s in range(sample_size):
			if membershipList[m][s]!=0:
				countCluster=countCluster+1
				break
	return countCluster

# Function to calculate Maximum distance between cluster
def MaxClusterDistance(M,S):

	Model=M
	sampleData=S

	MaxCD=0.0
	for i in range(SDmax-1):
		if not Check_Empty_Model(Model,i):
			for j in range(i+1, SDmax):
				if not Check_Empty_Model(Model,j):
					tempDis=0.0
					for d in range(dimension):
						if (float(Model[i][d])!=0.0 and float(Model[j][d]!=0.0)):
							tempDis=tempDis+ (float(Model[i][d])-float(Model[j][d]))
						else:
							tempDis=tempDis+ (float(Model[i][d])-0.0)
					if MaxCD<tempDis:
						MaxCD=tempDis
	return MaxCD

# Function to calculate Minimum distance between cluster
def MinClusterDistance(M,S):

	Model=M
	sampleData=S

	MinCD=9999999.0
	for i in range(SDmax-1):
		if not Check_Empty_Model(Model,i):
			for j in range(i+1, SDmax):
				if not Check_Empty_Model(Model,j):
					tempDis=0.0
					for d in range(dimension):
						if (float(Model[i][d])!=0.0 and float(Model[j][d])!=0.0):
							tempDis=tempDis+ (float(Model[i][d])-float(Model[j][d]))
						else:
							tempDis=tempDis+ (float(Model[i][d])-0.0)
					if tempDis<MinCD:
						MinCD=tempDis
	if MinCD==9999999.0:
		return 0.0
	else:
		return MinCD

# Function to calculate membership degree of each data
def membershipDegree(M,S):

	sampleData=S
	Model_mem=[]

	for i in range(SDmax):
		Model_mem.append([])
		for j in range(dimension):
			Model_mem[i].append(M[i][j])

	for i in range(SDmax):
		#vectorize
		for j in range(sample_size):
			membershipList[i][j]=0

	for s in range(sample_size):
		currentCluster=0
		dis=9999999.0
		for m in range(SDmax):
			newDis=0.0
			if not Check_Empty_Model(Model_mem,m):
				newDis=0.0
				for d in range(dimension):
					if float(Model_mem[m][d])==0.0:
						newDis=newDis+ (float(sampleData[s][d])-0.0)
					else:
						newDis=newDis+ (float(sampleData[s][d])-float(Model_mem[m][d]))
				if abs(newDis) < dis:
					dis = abs(newDis)
					currentCluster=m
		for n in range(SDmax):
			if n==currentCluster and dis != 9999999.0:
				membershipList[n][s]=1

#function to calculate distance between data and standard mean
def data_meanDistance(M,S):

	Model=M
	sampleData=S

	E1_dis=0.0
	for s in range(sample_size):
		for d in range(dimension):
			E1_dis=E1_dis+ (float(sampleData[s][d])-0.0)
	return E1_dis

#function to calculate distance between a data and its corrosponding cluster center
def data_clusterDistance(M,S):

	Model=M
	sampleData=S

	E_c_dis=0.0
	for m in range(SDmax):
		if not Check_Empty_Model(Model,m):
			for s in range(sample_size):
				DisCal=0
				for d in range(dimension):
					if float(Model[m][d])==0.0:
						DisCal=DisCal+ (float(sampleData[s][d])-0.0)
					else:
						DisCal=DisCal+ (float(sampleData[s][d])-float(Model[m][d]))
				E_c_dis=E_c_dis+membershipList[m][s]*DisCal
	return E_c_dis


#code for One_Child()
def OneChild(M ,G,S, SDmax):
	weight=0
	
	Model_child=[]
	genotype_child=[]

	for i in range(SDmax):
		Model_child.append([])
		genotype_child.append([])
		for j in range(dimension):
			Model_child[i].append(M[i][j])
			genotype_child[i].append(G[i][j])
	
	sampleData=S
	clusterRow=[]
	nonClusterRow=[]
	#print 'child',M
	for i in range(SDmax):
		for j in range(dimension):
			weight=weight + genotype_child[i][j]
	#print 'weight Before', weight

	if weight==SDmax:
		r1=(random.randint(0,SDmax-1))
		r2=(random.randint(0,dimension-1))
		while genotype_child[r1][r2] == 0:
			r1=(random.randint(0,SDmax-1))
			r2=(random.randint(0,dimension-1))
		genotype_child[r1][r2]=genotype_child[r1][r2]-1
		if genotype_child[r1][r2]==0:
			Model_child[r1][r2]=0.0

	rSample=(random.randint(0,sample_size-1))
	rdim=(random.randint(0,dimension-1))

	for m in range(SDmax):
		geneCount=0
		for d in range(dimension):
			geneCount=geneCount+genotype_child[m][d]
		if geneCount>0:
			clusterRow.append(m)
		else:
			nonClusterRow.append(m)

	prob=random.random()

	if weight==0:
		weight=1

	if prob <=(1/(weight+0.0)) or (0.0 <prob<.25) :
		point=(random.randint(0,len(nonClusterRow)-1))
		c=nonClusterRow[point]
	else:
		point=(random.randint(0,len(clusterRow)-1))
		c=clusterRow[point]

	genotype_child[c][rdim]=genotype_child[c][rdim]+1
	Model_child[c][rdim]=sampleData[rSample][rdim]

	return Model_child, genotype_child

#Cluster wrt overall data(membership)
def BUILDSUBSPACECLUSTERS(data, M):

	Model=M
	allData=data

	for i in range(SDmax):
		#vectorise
		for j in range(dataCount):
			membershipListAllData[i][j]=0

	for s in range(dataCount):
		currentCluster=0
		dis=9999999.0
		for m in range(SDmax):
			if not Check_Empty_Model(Model,m):
				newDis=0.0
				for d in range(dimension):
					if float(Model[m][d])==0.0:
						newDis=newDis+ (float(allData[s][d])-0.0)
					else:
						newDis=newDis+(float(allData[s][d])-float(Model[m][d]))
				if abs(newDis) < dis:
					dis = abs(newDis)
					currentCluster=m
				#eliminate the code below
		for n in range(SDmax):
			if n==currentCluster and dis != 9999999.0:
				membershipListAllData[n][s]=1

#Function to calculate non zero cluster membership
def Check_Empty_ModelAllData():
	for m in range (SDmax):
		for d in range(dataCount):
			if membershipListAllData[m][d]!=0:
				membershipListAllDataCopy.append(membershipListAllData[m])
				break
	#print  (membershipListAllDataCopy)

def FinalcountCluster():
	cluster=0
	for m in range (SDmax):
		for s in range(dataCount):
			#vectorise
			if membershipListAllData[m][s]!=0:
				cluster=cluster+1
				break
	return cluster

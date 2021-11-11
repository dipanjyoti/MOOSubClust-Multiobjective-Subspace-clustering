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

from util import *
from MOO import *
from objectives import *
from evaluation import *
from parameters import *




# *************************************************************  MAIN BODY OF PROGRAM  ***********************************************************************



readfile.close()
normalizedData()

old=[dataCount]

for i in range(0,sample_size):
	while True:
		k=(random.randint(0,dataCount-1))
		#print 'k=',k
		if k not in old:
			old.append(k)
			break
	sampleData.append(mydata_list_of_list[k])

#initializing an empty model(Clusters Centers coordinate)
Model=[] 
for i in range(SDmax):
    Model.append([])
	#Model[i]=np.zeros(dimension,dtype=float)
    for j in range(dimension):
         Model[i].append(0)

genotype=[] #defines the no of data in each dimension
for i in range(SDmax):
    genotype.append([])
	#genotype[i]=np.zeros(dimension,dtype=float)
    for j in range(dimension):
         genotype[i].append(0)

Model_new=[]
genotype_new=[]

population=[]
population_new=[]
population_final=[]
sol=[]

for i in range(crowding_size):
	sol.append((0,0))
for i in range(crowding_size):
	population.append((0,0))
for i in range(crowding_size):
	population_new.append((0,0))
for i in range(2*crowding_size):
	population_final.append((0,0))

for i in range(SDmax): #List to put Crisp membership degree
    membershipList.append([])
    for j in range(sample_size):
    	membershipList[i].append(0)

for i in range(SDmax): #List to put Crisp membership degree
    membershipListAllData.append([])
    for j in range(dataCount):
    	membershipListAllData[i].append(0)


def GenerateSolution(Fonts):
    global F1_len
    k=crowding_size
    distance=[]
    i=0
    x=0
    F=[]
    F=Fonts
    while True:
	    if i>=len(F):
		    return
	    else:
		    if len(F[i])>k:
			    distance=crowding_distance(F[i])
			    e=dict()
			    
			    for j in range(len(F[i])):
				    e[distance[j]]=j

			    distance.sort(reverse=True)
	
			    for j in range (k):
				    sol[x]=F[i][e.get(distance[j])]
				    x=x+1
			    break
		    else:
			    for j in range(len(F[i])):
				    sol[x]=F[i][j]
				    x=x+1
			    k=k-len(F[i])
	    if i==0:
		    F1_len=len(F[0])
	    i=i+1


#Repeat while lpooop

Iter=0
while Iter<crowding_size:

	Modelnew=[]
	Modelnew1=[]
	genotypenew=[]
	genotypenew1=[]
	Modelnew, genotypenew=OneChild(Model,genotype,sampleData,SDmax)
	for i in range(init_value):
		Modelnew, genotypenew=OneChild(Modelnew,genotypenew,sampleData,SDmax)

	#print 'ModelNew',Modelnew
	z=0
	while z<z_value:
		
		Modelnew1, genotypenew1=OneChild(Modelnew,genotypenew,sampleData,SDmax)
		membershipDegree(Modelnew,sampleData)
		XB_old=Calculate_XB(Modelnew,sampleData)
		PBM_old=Calculate_PBM(Modelnew,sampleData)
		membershipDegree(Modelnew1,sampleData)
		XB_New=Calculate_XB(Modelnew1,sampleData)
		PBM_New=Calculate_PBM(Modelnew1,sampleData)
		if (XB_old > XB_New and PBM_old < PBM_New) or (XB_old < XB_New and PBM_old > PBM_New):
			flag=(random.randint(0,1))
			if flag==0:
				Modelnew=Modelnew1
				genotypenew=genotypenew1
		elif XB_old >= XB_New and PBM_old >= PBM_New:
			Modelnew=Modelnew1
			genotypenew=genotypenew1
		z=z+1
	membershipDegree(Modelnew,sampleData)
	XB= Calculate_XB(Modelnew,sampleData)
	PBM= Calculate_PBM(Modelnew,sampleData)
	population[Iter]=(XB,PBM) 
	Model_dict[Iter]=Modelnew
	genotype_dict[Iter]=genotypenew
	randSample=(random.randint(0,sample_size-1)) # changing sample data one at a time
	randData=(random.randint(0,dataCount-1))
	sampleData[randSample]=mydata_list_of_list[randData]
	Iter=Iter+1

while Iteration<NoOfIteration:

    randSample=(random.randint(0,sample_size-1)) # changing sample data one at a time
    randData=(random.randint(0,dataCount-1))
    sampleData[randSample]=mydata_list_of_list[randData]
    Model_dict_new={}
    genotype_dict_new={}
    Iter=0
    #print 'Iteration=', Iteration
    if(Iteration%200==0):
    	print 'Iteration=', Iteration

    while Iter<crowding_size:
		
	    Model_new, genotype_new=OneChild(list(Model_dict.values())[Iter],list(genotype_dict.values())[Iter],sampleData,SDmax)
	    membershipDegree(Model_new,sampleData)
	    XB= Calculate_XB(Model_new,sampleData)
	    PBM= Calculate_PBM(Model_new,sampleData)
	    #print 'Model_new', Model_new
	    population_new[Iter]=(XB,PBM)
	    Model_dict_new[Iter]=Model_new
	    genotype_dict_new[Iter]=genotype_new
	    Iter=Iter+1

    for k in range(crowding_size): #combining population and population_new

	    population_final[k]=population[k]
	    Model_final_dict[k]=Model_dict[k]
	    genotype_final_dict[k]=genotype_dict[k]
	    population_final[k+crowding_size]=population_new[k]
	    Model_final_dict[k+crowding_size]=Model_dict_new[k]
	    genotype_final_dict[k+crowding_size]=genotype_dict_new[k]
		   
    #print (len(population_final))
    F=non_dominating(population_final)
    GenerateSolution(F)
    for k in range(crowding_size):

        index=population_final.index(sol[k])
        Model_dict[k]=Model_final_dict[index]
        genotype_dict[k]=genotype_final_dict[index]
        population[k]=population_final[index]

    Iteration=Iteration+1



#######CLUSTER EVALUATION######

def sameClusterPair(data1,data2,Level):
	bit=0
	for i in range (len(Level)):
		if data1 in Level[i] and data2 in Level[i]:
			bit=1
			break
	if bit==1:
		return 1
	else:
		return 0

#print ('True Cluster=',ClusterTrue)
#print ('Pred Cluster=',ClusterPred)

accuracy_list=[]
for model in range(crowding_size):

	Model=Model_dict[model]
	BUILDSUBSPACECLUSTERS(mydata_list_of_list, Model)
	Check_Empty_ModelAllData()

	PredLevel=[]
	for s in range (dataCount):
		for m in range (len(membershipListAllDataCopy)): # SDmax can be replace by len(membershipListAllDataCopy)
			if membershipListAllDataCopy[m][s]==1:
				PredLevel.append(m)
				break
	Total_Cluster=len(membershipListAllDataCopy)
	EntroProb=[]
	ClusterPred=[]
	ClusterTrue=[]
	for i in range(len(membershipListAllDataCopy)):
		ClusterPred.append([])

	for i in range(NoOfClasses):
		ClusterTrue.append([])

	for i in range(dataCount):
		ClusterPred[PredLevel[i]].append(i)
		ClusterTrue[trueLevel[i]].append(i)

	XY=0.0
	for i in range (0,dataCount-1):
		for j in range (i+1,dataCount):
			p=sameClusterPair(i,j,ClusterTrue)
			q=sameClusterPair(i,j,ClusterPred)
			if p==q:
				XY=XY+1

	n=(dataCount*(dataCount-1))/2
	Accuracy=(XY/(n+0.0))
	accuracy_list.append(Accuracy)

print ('All_Accuracy=',accuracy_list)

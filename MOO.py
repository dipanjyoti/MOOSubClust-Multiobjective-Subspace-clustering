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
from util import *


#Code for crowding distance

def find_minimize(sol1,sol2):
	return all(i <= j for i, j in zip(sol1,sol2))

def compliment(a,b):
	new_list=[]
	for i in b:
		if i!=a:
			new_list.append(i)
	return tuple(new_list)


def Sort(F,i):
	return (sorted(F,key=lambda x:x[i]))
def Max(F,i):
	return (max(F,key=lambda x:x[i])[i])
def Min(F,i):
	return (min(F,key=lambda x:x[i])[i])

def crowding_distance(Font):
	F=[]
	F=Font
	distance=[]
	for i in range(len(F)):
		distance.append(0)
	noOfObjectives=2

	for i in range (noOfObjectives):
		F_new=Sort(F,i)
		if Max(F_new,i)==Min(F_new,i):
			continue
		distance[0]=99999.0
		distance[len(F)-1]=99999.0
		for j in range (1,len(F)-1):
			distance[j]+=(float)(F_new[j+1][i]-F_new[j-1][i])/((Max(F_new,i)-Min(F_new,i))+0.0)
	return distance

# def GenerateSolution(Fonts):
#     global F1_len
#     k=crowding_size
#     distance=[]
#     i=0
#     x=0
#     F=[]
#     F=Fonts
#     while True:
# 	    if i>=len(F):
# 		    return
# 	    else:
# 		    if len(F[i])>k:
# 			    distance=crowding_distance(F[i])
# 			    e=dict()
			    
# 			    for j in range(len(F[i])):
# 				    e[distance[j]]=j

# 			    distance.sort(reverse=True)
	
# 			    for j in range (k):
# 				    sol[x]=F[i][e.get(distance[j])]
# 				    x=x+1
# 			    break
# 		    else:
# 			    for j in range(len(F[i])):
# 				    sol[x]=F[i][j]
# 				    x=x+1
# 			    k=k-len(F[i])
# 	    if i==0:
# 		    F1_len=len(F[0])
# 	    i=i+1


def non_dominating(P):
	population1=[]
	population1=P

	F_one_set=[]
	for i in range(2*crowding_size):
		F_one_set.append([])
	ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=population1)
	for i in range(2*crowding_size):
		F_one_set[ndr[i]].append(population1[i])
	F_one_set2 = [x for x in F_one_set if x != []]
	return F_one_set2


def non_dominating_(P):
	population=tuple(P)
	rank_sol=[]
	rank_dict=OrderedDict()
	current=0

	for i in population:
		rank_sol.append((i,current))

	rank_dict= OrderedDict({k: v for k, v in rank_sol})

	solution_space=OrderedDict()
	n_sol_space=OrderedDict()
	n_sol=0
	for i in population:
		n_sol_space.update({i:n_sol})

	F_one_set=[]
	F_one_set.append([])

	for i in population:
		F_one_set.append([])

	F_one=[]

	current_sol=population[0]
	for sol in population:
		solution=[]
		for sol_dash in compliment(sol,population):
			if find_minimize(sol,sol_dash):
				if sol_dash not in solution:
					solution.append(sol_dash)
			elif find_minimize(sol_dash,sol):
				n_sol_space[sol]=n_sol_space[sol]+1
		if n_sol_space[sol]==0:
			rank_dict[sol]=1
			F_one.append(sol)
		solution_space.update({sol:tuple(solution)})

	F_one_set[0]=F_one

	i=1
	while(len(F_one_set[i-1])!=0):
		Q=[]
		for soll in F_one_set[i-1]:
			for soll_dash in solution_space[soll]:
				n_sol_space[soll_dash]=n_sol_space[soll_dash]-1
				if n_sol_space[soll_dash]==0:
					rank_dict[soll_dash]=i+1
					Q.append(soll_dash)

		i=i+1
		F_one_set[i-1]=Q

	return F_one_set


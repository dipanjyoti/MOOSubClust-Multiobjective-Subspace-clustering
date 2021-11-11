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


#function to calculate PBM index
def Calculate_PBM(M,S):
	Model=M
	sampleData=S

	Num_cluster=Cal_Noof_Cluster()
	E1_distance=data_meanDistance(Model,sampleData)
	E_c_distance=data_clusterDistance(Model,sampleData)
	MCD=MaxClusterDistance(Model,sampleData)

	if Num_cluster==0 or E_c_distance==0.0 or MCD==0.0:
		PBM_value=0.0
	else:
		PBM_value=(E1_distance*E1_distance*MCD*MCD)/((Num_cluster*Num_cluster*E_c_distance*E_c_distance)+0.0)

	if PBM_value==0.0:
		return 0.0
	else:
		return (float)(1/(PBM_value+0.0))

#function to calculate XB index
def Calculate_XB(M,S):
	Model=M
	sampleData=S

	compactDistance=data_clusterDistance(Model,sampleData)
	mCD=MinClusterDistance(Model,sampleData)
	if mCD==0.0:
		XB_value=0.0
	else:
		XB_value=(compactDistance*compactDistance)/((sample_size*mCD*mCD)+0.0)
	return XB_value

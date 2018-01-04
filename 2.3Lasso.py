# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 01:54:31 2017

@author: Jasdeep
"""

import matplotlib.pyplot as plotting
import numpy as np
import csv
from sklearn.model_selection import KFold


inputfile = open('housing_X_train.csv', 'r')
inputreader = csv.reader(inputfile,delimiter=',')
dataset = list(inputreader)
data_array = np.array(dataset).astype("float")

#print(data_array[:,0])

outputFile = open('housing_y_train.csv', 'r')
outputreader = csv.reader(outputFile, delimiter = ',')
resultset = list(outputreader)
result_array = np.array(resultset).astype("float")

col=data_array.shape[1]
row=data_array.shape[0]

def SignZ(beta, labda):
    if abs(beta)>labda:
        if beta>0:
            return beta-labda
        elif beta<0:
            return beta+labda
    else:
        return 0
    
def mean_square_error(X,weight_array,result_array,number_of_dataPts):
    A=(np.dot(X,weight_array)).reshape(X.shape[0],1)-result_array ## predicted - actual value giver vector
    B=np.dot(A.T,A) # gives magnitude of above vector
    return B/number_of_dataPts
    
def PreprocessStep(data_array,weight_array):
    PreArray=np.zeros(data_array.shape[0])
    for k in range(0,col):
        PreArray+=data_array[:,k]*weight_array[k] #multiplying each column with its corresponding weight    
    return PreArray
 
#########################################################################
class LabdaNode():
    def __init__(self,labda,meanSqErr):
        self.labda = labda
        self.meanSqErr = meanSqErr 
        
No_of_folds = 10
CV = KFold(n_splits=No_of_folds)

TrainingDictforLabdaWt = {}

Final_list_for_labda = list() 
Final_list_index = 0

def standardising_data(data_set):
    data_set = (data_set - np.mean(data_set, axis=0)) / np.std(data_set, axis=0)
    return data_set
        

def lassoRegression(data_array,result_array,labda):
    tol=10**-3
    weight_array=np.zeros(col)
    oldWtVector=np.zeros(col)
    Preprocess=PreprocessStep(data_array,weight_array)
    row=data_array.shape[0]
    while(True):
        oldWtVector=np.copy(weight_array)
        for j in range(0,col):
            currentColumn=data_array[:,j]
            Preprocess= Preprocess - weight_array[j]*currentColumn
            C=Preprocess.reshape(row,1) #subtracting current column
            C1= np.subtract(C,result_array)
            #print("C1 shape",C1.shape)
            Numerator=-(np.dot(C1.T,currentColumn))
            Denominator=np.dot(currentColumn.T,currentColumn)
            C2=Numerator/Denominator
            lambdastar=labda/Denominator
            updatedBeta=SignZ(C2,lambdastar)
            #print("UpdatedBeta",updatedBeta)
            weight_array[j]=updatedBeta #weight not getting updated right
            Preprocess=np.add(Preprocess,currentColumn*updatedBeta)
        updatedwtVector=weight_array
        
        if abs(np.dot(updatedwtVector,updatedwtVector)-np.dot(oldWtVector,oldWtVector))<tol:
            break
        #print("number of zeros at labda %d is %d" %(labda,updatedwtVector.shape[0]-np.count_nonzero(updatedwtVector)))
    return updatedwtVector
    #print("Final number of zeros" ,updatedwtVector.shape[0]-np.count_nonzero(updatedwtVector))
'''for labda in range (0,110,10):
    weightvector=lassoRegression(data_array,result_array,labda)
    no_of_features=weightvector.shape[0]
    no_of_zeros=no_of_features-np.count_nonzero(weightvector)
    print("number of zeros at lambda %d is %d and mag. of wt vector is %f" %(labda,no_of_zeros,np.dot(weightvector,weightvector)))'''
    

for labda in range(0,110,10):
    LabdaListindex = 0
    Validation_error_list = list()
    for training_index, validation_index in CV.split(data_array):
        copydata_array = np.copy(data_array)
        iListIndex = list(copydata_array)
        
        copyresult_array = np.copy(result_array)
        oListIndex = list(copyresult_array)
        
        
        iListValidation = list()
        oListValidation = list()
        
        for i in validation_index:
            iListValidation.append(iListIndex[i])
            oListValidation.append(oListIndex[i])
        
        validationIArray = np.asarray(iListValidation)
        validationOArray = np.asarray(oListValidation)
        number_of_dataPts = validationIArray.shape[0]
        
        ilistTraining = list() 
        oListTraining = list()
        for i in training_index:
            ilistTraining.append(iListIndex[i])
            oListTraining.append(oListIndex[i])
            
        traindata_array = np.asarray(ilistTraining)
        trainresult_array = np.asarray(oListTraining)
        
        
        coefficients = lassoRegression(traindata_array,trainresult_array,labda)
        
        MSError = mean_square_error(validationIArray,coefficients,validationOArray,number_of_dataPts)
        #print("AT lambda %d Error is %f" % (labda,MSError))
        Validation_error_list.insert(LabdaListindex,MSError)
        LabdaListindex+=1;
    AvgValErLabda = np.mean(Validation_error_list)
    Node = LabdaNode(labda,AvgValErLabda)
    Final_list_for_labda.insert(Final_list_index,Node)
    Final_list_index+=1

CVx = []
CVy = []
for i in range(len(Final_list_for_labda)):
    val = Final_list_for_labda[i]
    CVx.insert(i,val.labda)
    CVy.insert(i,val.meanSqErr)
    print("At Lambda %d ,mean CV error %f" %(val.labda, val.meanSqErr))


# training error Calculation

TrainX = []
TrainY = []
Index = 0
PrcntNonZeroX=[]
PrcntNonZeroY=[]
for k in range(0,110,10):
    Coefficient = lassoRegression(data_array,result_array,k)
    MSErrTraining = mean_square_error(data_array,Coefficient,result_array,data_array.shape[0])
    TrainX.insert(Index,k)
    TrainY.insert(Index,MSErrTraining[0][0])
    
    Index+=1
    TrainingDictforLabdaWt[k] = Coefficient
    
    Percent_of_NZero=(np.count_nonzero(Coefficient)/Coefficient.shape[0])*100
    PrcntNonZeroX.insert(Index,k)
    PrcntNonZeroY.insert(Index,Percent_of_NZero)
    print("At Lambda %d,Training Error is %f "% (k,MSErrTraining))
    print("Percentage of non-zero %d is %f "% (k,Percent_of_NZero))


#Reading of Test Data:
    
oFileTest = open('housing_y_test.csv', 'r')
oreaderTest = csv.reader(oFileTest, delimiter = ',')
oDataTest = list(oreaderTest)
oArrayTest = np.array(oDataTest).astype("float")
    
ifileTest = open('housing_X_test.csv', 'r')
ireaderTest = csv.reader(ifileTest,delimiter=',')
iDataTest = list(ireaderTest)
iArrayTest = np.array(iDataTest).astype("float")

#iArrayTest = np.c_[iArrayTest,np.ones(iArrayTest.shape[0])]

TestX = []
TestY = []
Index = 0
for i in range(0,110,10):
    MSErrTest = mean_square_error(iArrayTest,TrainingDictforLabdaWt[i],oArrayTest,iArrayTest.shape[0])
    TestX.insert(Index,i)
    TestY.insert(Index,MSErrTest[0][0])
    Index+=1
    print("At Lambda %d,Test Error is %f "% (i,MSErrTest))
    

graph = plotting.figure()

plotCV = graph.add_subplot(221)
plotCV.plot(CVx, CVy, 'r-')
plotCV.set_xlabel("Value of Lambda")
plotCV.set_ylabel("CrossValidation error")

plotTrain = graph.add_subplot(222)
plotTrain.plot(TrainX, TrainY, 'r-')
plotTrain.set_xlabel("Value of Lambda")
plotTrain.set_ylabel("Training error")

plotTest = graph.add_subplot(223)
plotTest.plot(TestX, TestY, 'r-')
plotTest.set_xlabel("Value of Lambda")
plotTest.set_ylabel("Test error")    

plotTest = graph.add_subplot(224)
plotTest.plot(PrcntNonZeroX, PrcntNonZeroY, 'r-')
plotTest.set_xlabel("Value of Lambda")
plotTest.set_ylabel("Percent of Non zero")
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 20:30:46 2017

@author: Karan Vijay Singh
"""
import matplotlib.pyplot as plotting
import numpy as np
import csv
from sklearn.model_selection import KFold
import random

oFile = open('housing_y_train.csv', 'r')
oreader = csv.reader(oFile, delimiter = ',')
resultset = list(oreader)
result_array = np.array(resultset).astype("float")

ifile = open('housing_X_train.csv', 'r')
ireader = csv.reader(ifile,delimiter=',')
dataset = list(ireader)
data_array = np.array(dataset).astype("float")

#class with two members lambda and mean square error
class LabdaNode():
    def __init__(self,labda,meanSqErr):
        self.labda = labda
        self.meanSqErr = meanSqErr 


No_of_folds = 10
CV = KFold(n_splits=No_of_folds)

TrainingDictforLabdaWt = {}

   

def standardising_data(data_set):
    data_set = (data_set - np.mean(data_set, axis=0)) / np.std(data_set, axis=0)
    return data_set
#Function to implement Ridge Regression
def ridge_regression(data_array,result_array,labda):  
    X=data_array
    #number_of_dataPts=X.shape[0] ##no of rows
    features=X.shape[1] ##no. of columns i.e features
    identityM = np.identity(features, dtype = np.float64)
    identityM[features-1, features-1] = 0
    A=np.dot(X.transpose(),X)+labda*identityM
    B=np.dot(X.transpose(),result_array)
    weight_array = np.linalg.solve(A, B)
    return weight_array
    
#Function to calculate mean square error
def mean_square_error(X,weight_array,result_array,number_of_dataPts):
    A=np.dot(X,weight_array)-result_array ## predicted - actual value giver vector
    B=np.dot(A.T,A) # gives magnitude of above vector
    return B/number_of_dataPts

    
Final_list_for_labda = list() 
Final_list_index = 0

#Choosing a Random row and multiplying the row by 10^6 and Corresonding y by 10^3
RandomValue=random.randint(0,data_array.shape[0])
data_array[RandomValue]=data_array[RandomValue]*1000000
result_array[RandomValue]=result_array[RandomValue]*1000

#Addidng ones for intercept at the end of data set
data_array = np.c_[data_array,np.ones(data_array.shape[0])] 

#Applying kfold and calculating cross validation error for different lambdas
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
        
        
        coefficients = ridge_regression(traindata_array,trainresult_array,labda)
        
        MSError = mean_square_error(validationIArray,coefficients,validationOArray,number_of_dataPts)
        #print("At lambda = %d,MSE is %f" % (labda,MSError))
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
    


#Training error to calculate

TrainX = []
TrainY = []
Index = 0
PrcntNonZeroX=[]
PrcntNonZeroY=[]
for k in range(0,110,10):
    Coefficient = ridge_regression(data_array,result_array,k)
    MSErrTraining = mean_square_error(data_array,Coefficient,result_array,data_array.shape[0])
    TrainX.insert(Index,k)
    TrainY.insert(Index,MSErrTraining[0][0])
    Index+=1
    TrainingDictforLabdaWt[k] = Coefficient
    Percent_of_NZero=(np.count_nonzero(Coefficient)/Coefficient.shape[0])*100
    PrcntNonZeroX.insert(Index,k)
    PrcntNonZeroY.insert(Index,Percent_of_NZero)
    print("At Lambda %d,Training Error is %f "% (k,MSErrTraining))
    #print("Percantage of non-zero %d is %f "% (k,Percent_of_NZero))



#Test Data Read   
oFileTest = open('housing_y_test.csv', 'r')
oreaderTest = csv.reader(oFileTest, delimiter = ',')
oDataTest = list(oreaderTest)
oArrayTest = np.array(oDataTest).astype("float")
    
ifileTest = open('housing_X_test.csv', 'r')
ireaderTest = csv.reader(ifileTest,delimiter=',')
iDataTest = list(ireaderTest)
iArrayTest = np.array(iDataTest).astype("float")

#Addidng ones for intercept at the end of test set
iArrayTest = np.c_[iArrayTest,np.ones(iArrayTest.shape[0])]

#Test Error
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
#Plotting Cross Validation error
plotCV = graph.add_subplot(221)
plotCV.plot(CVx, CVy, 'r-')
plotCV.set_xlabel("Value of Lambda")
plotCV.set_ylabel("CrossValidation error")
#Plotting Training Error
plotTrain = graph.add_subplot(222)
plotTrain.plot(TrainX, TrainY, 'r-')
plotTrain.set_xlabel("Value of Lambda")
plotTrain.set_ylabel("Training error")

'''plotTest = graph.add_subplot(223)
plotTest.plot(PrcntNonZeroX, PrcntNonZeroY, 'r-')
plotTest.set_xlabel("Value of Lambda")
plotTest.set_ylabel("Percent of Non zero")'''  
#Plotting Test Error
plotTest = graph.add_subplot(223)
plotTest.plot(TestX, TestY, 'r-')
plotTest.set_xlabel("Value of Lambda")
plotTest.set_ylabel("Test error")    
    
    
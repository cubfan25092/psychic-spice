# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as pl

EPOCH_COUNT = 100
TRAINING_COUNT = 1000

def catVector(name):
    retVal = []  
    categories = []
    categories.append(['Private','Self-emp-not-inc','Self-emp-inc','Federal-gov',
                       'Local-gov','State-gov','Without-pay','Never-worked'])
    categories.append(['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 
                       'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', 
                       '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'])
    categories.append(['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 
                       'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
    categories.append(['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 
                       'Exec-managerial','Prof-specialty', 'Handlers-cleaners', 
                       'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 
                       'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
    categories.append(['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']) 
    categories.append(['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']) 
    categories.append(['Female', 'Male']) 
    categories.append(['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 
                       'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 
                       'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 
                       'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 
                       'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 
                       'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'])  
                 
    for i in categories:
        if name in i:
            index = i.index(name)
            return index
         
    print name
    return -1
    
def runEpoch(currLambda, trainingSet, w, b, epochNum):
    stepLength = 1.0/((.01* float(epochNum)+50.0))
    rowLength = np.shape(trainingSet)[1]
    
    for i in xrange(TRAINING_COUNT):
        row = trainingSet[np.random.randint(trainingSet.shape[0])]
        yk = row[rowLength-1]
        xk = row[0:rowLength-1]
        #print yk
        if yk*(np.dot(w,xk) + b) >= 1:
            w = w - (stepLength * (currLambda * w))
        else:
            w = w - (stepLength * ((currLambda * w) - (yk * xk)))
            b = b - (stepLength * ((-1) * yk))
            
            
    return w, b
    
def lsvm(parsedData):
    rowLength = np.shape(parsedData[0])[0]
    

    for x in xrange(rowLength-1):
        
        col = parsedData[:, x]
        if (np.std(col) != 0.0):
            col = (col - np.mean(col))/np.std(col)
            parsedData[:,x] = col
      
    np.random.shuffle(parsedData)
    
    validationSet = parsedData[:1000]
    testSet = parsedData[1000:6000]
    trainingSet = parsedData[6000:]
    
    print np.shape(validationSet)
    print np.shape(testSet)
    print np.shape(trainingSet)
    
    w = np.zeros((EPOCH_COUNT,rowLength-1))
    b = np.zeros(EPOCH_COUNT)
    currLambda = 1e-7
    x = np.linspace(1,EPOCH_COUNT,EPOCH_COUNT)
    wMag = np.zeros(EPOCH_COUNT)
    errorVals = np.zeros(EPOCH_COUNT)
    print float(np.shape(validationSet)[0])
    
    for i in xrange(8):
        print currLambda
        for j in xrange(EPOCH_COUNT):
            if j == 0:
                w[j], b[j] = runEpoch(currLambda, trainingSet, w[0], b[0], j)
            else:
                w[j], b[j] = runEpoch(currLambda, trainingSet, w[j-1], b[j-1], j)
                
            wMag[j] = np.sqrt(np.dot(w[j],w[j]))
            
            currCount = 0
            for k in validationSet:
                
                result = np.dot(w[j], k[:np.size(k)-1]) + b[j]
                if (result > 0 and  k[np.size(k)-1] > 0) or (result < 0 and  k[np.size(k)-1] < 0):
                    currCount += 1
                    
            
            errorVals[j] = float(currCount) / float((np.shape(validationSet)[0]))

          
        pl.figure(0)
        pl.plot(x, wMag, label=str(currLambda))
        pl.figure(1)
        pl.plot(x, errorVals, label=str(currLambda))
        currLambda *= 10

    pl.figure(1)
    pl.ylim([0,1])    
    pl.legend()
    pl.show()
            
    
def parseData(rawData, option):
    
    if option == 1:    
        numericData = rawData[:,[0,2,4,10,11,12,14]]
        for x in numericData:
            if x[6] == '<=50K':
                x[6] = '-1'
            else:
                x[6] = '1'
                
        return numericData.astype(np.float)
        
    else:
        retVal = np.zeros((np.shape(rawData)[0],106))
        retVal[:,0] = rawData[:,0]
        retVal[:,9] = rawData[:,2]
        retVal[:,26] = rawData[:,4]
        retVal[:,61] = rawData[:,10]
        retVal[:,62] = rawData[:,11]
        retVal[:,63] = rawData[:,12]
                
        for row in xrange(np.shape(rawData)[0]):
            if '?' not in rawData[row]:
                if rawData[row][14] == '<=50K':
                    retVal[row][105] = -1
                else:
                    retVal[row][105] = 1
                    
                
                retVal[row][1 + catVector(rawData[row][1])] = 1 
                retVal[row][10 + catVector(rawData[row][3])] = 1
                retVal[row][27 + catVector(rawData[row][5])] = 1
                retVal[row][34 + catVector(rawData[row][6])] = 1
                retVal[row][48 + catVector(rawData[row][7])] = 1
                retVal[row][54 + catVector(rawData[row][8])] = 1
                retVal[row][59 + catVector(rawData[row][9])] = 1
                retVal[row][64 + catVector(rawData[row][13])] = 1
                
            
        return retVal.astype(np.float)
            
        

def main():
    rawData = np.loadtxt('adult.data', dtype = str, delimiter = ", " )
    parsedData = parseData(rawData, 1)
    lsvm(parsedData)


if __name__ == "__main__":
    main()
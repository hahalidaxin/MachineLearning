import numpy as np
import operator
import sys
# sys.path.append('.')
import os
import matplotlib.pyplot as plt


def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group ,labels


def classify0(inX,dataSet,lables,k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet
    sqDifffMat = diffMat ** 2
    sqDistances = sqDifffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistances = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = lables[sortedDistances[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = np.zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return
    fr = open(filename)
    index = 0
    numoftype = 0
    flagoftype = {}
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        if flagoftype.get(listFromLine[-1]) is None:
            flagoftype[listFromLine[-1]] = numoftype
            numoftype += 1
        labelvector = flagoftype[listFromLine[-1]]
        classLabelVector.append(labelvector)
        index += 1
    return returnMat,classLabelVector


def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        linestr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(linestr[j])
    return returnVect


def normdata(dataSet):
    minVal = dataSet.min(0)
    maxVal = dataSet.max(0)
    ranges = maxVal - minVal
    return (dataSet-np.tile(minVal,(dataSet.shape[0],1)))/np.tile(ranges,(dataSet.shape[0],1))


def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = os.listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print ("\nthe total number of errors is: %d" % errorCount)
    print ("\nthe total error rate is: %f" % (errorCount/float(mTest)))


'''
def plt_show():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingMat[:, 1], datingMat[:, 2], 15.0*np.array(datingLabel), 15.0*np.array(datingLabel))
    plt.show()
'''

if __name__=='__main__':
    handwritingClassTest()
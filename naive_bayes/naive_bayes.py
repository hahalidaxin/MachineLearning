import numpy as np


def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec


def createVocabList(dataList):
    ans = set()
    for data in dataList:
        ans = ans | set(data)
    return list(ans)


def setOWords2Vec(vocabList,inputSet):
    ans = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            ans[vocabList.index(word)] = 1
        else :
            print("word %s is not in vocablist"%word)
    return ans


def trainNB0(trainMattrix,trainCategory):

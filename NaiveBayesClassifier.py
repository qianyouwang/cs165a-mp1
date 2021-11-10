import pandas as pd
import numpy as np
import random
import sys


def trans_data(file):
    data = np.loadtxt(file, dtype=np.str_, encoding='utf-8')
    column = ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed',
              'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am',
              'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday', 'RainTomorrow']
    df = pd.DataFrame(data=data, columns=column)
    df = df.replace({',': ''}, regex=True)
    column1 = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am',
               'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
               'Temp9am', 'Temp3pm']
    for i in column1:
        df[i] = df[i].astype('float64')
    df1 = df[column1 + ['RainToday', 'RainTomorrow']]
    df1.loc[df1['RainToday'] == 'Yes', 'RainToday'] = 1
    df1.loc[df1['RainTomorrow'] == 'Yes', 'RainTomorrow'] = 1
    df1.loc[df1['RainToday'] == 'No', 'RainToday'] = 0
    df1.loc[df1['RainTomorrow'] == 'No', 'RainTomorrow'] = 0
    df1.to_csv('train.csv', index=False)

def loadcsv(file):
    dataset = pd.read_csv(file)
    dataset = dataset.values
    return dataset



def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


def mean(arr):
    return sum(arr)/float(len(arr))

def stdev(arr):
    avg=mean(arr)
    variance=sum([pow(x-avg,2) for x in arr])/float(len(arr)-1)
    return np.sqrt(variance)


#calculate mean and std classwise for each feature
def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries




def calculateProbability(x, mean, stdev):
    exponent = np.exp(-(np.power(x-mean,2)/(2*np.power(stdev,2))))
    return (1 / (np.sqrt(2*np.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities



#predicting the best label
def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

def main():
    # file='training.txt'
    trans_data(sys.argv[1])
    filename="train.csv"
    trainingSet=loadcsv(filename)
    # file = 'testing.txt'
    trans_data(sys.argv[2])
    filename = "train.csv"
    testSet = loadcsv(filename)
    summaries = summarizeByClass(trainingSet)
    # print(summaries)
    predictions = getPredictions(summaries, testSet)
    accuracy = getAccuracy(testSet, predictions)
    for i in predictions:
        print(int(i))
    # print('Accuracy:',round(accuracy))

main()


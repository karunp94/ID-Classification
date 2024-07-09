
import csv
import random
import math
import operator
import cv2


# calculation of euclidead distance
def calculateEuclideanDistance(variable1, variable2, length):
    distance = 0
    for x in range(length):
        distance += pow(variable1[x] - variable2[x], 2)
    return math.sqrt(distance)


# get k nearest neigbors
def kNearestNeighbors(list1, testInstance, k):
    distances = []
    length = len(testInstance)
    for x in range(len(list1)):
        dist = calculateEuclideanDistance(testInstance,
                list1[x], length)
        distances.append((list1[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


# votes of neighbors
def responseOfNeighbors(neighbors):
    all_possible_neighbors = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in all_possible_neighbors:
            all_possible_neighbors[response] += 1
        else:
            all_possible_neighbors[response] = 1
    sortedVotes = sorted(all_possible_neighbors.items(),
                         key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]



def loadDataset(
    filename,
    filename2,
    list1=[],
    list2=[],
    ):
    with open(filename) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(3):
                dataset[x][y] = float(dataset[x][y])
            list1.append(dataset[x])

    with open(filename2) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(3):
                dataset[x][y] = float(dataset[x][y])
            list2.append(dataset[x])


def main(training_data, test_data):
    list1 = []  # for loading the training data
    list2 = []  # for loading the test data
    loadDataset(training_data, test_data, list1, list2)
    classifier_prediction = []  # predictions
    k = 3  
    for x in range(len(list2)):
        neighbors = kNearestNeighbors(list1, list2[x], k)
        result = responseOfNeighbors(neighbors)
        classifier_prediction.append(result)
    return classifier_prediction[0]		

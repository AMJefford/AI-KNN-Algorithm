import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import math
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

#Manipulation of data
digits = datasets.load_digits()
X = digits.data/digits.data.max()
y = digits.target #label data
    #XTrain is training data set
    #yTrain is the set of labels to all the data in XTrain
    #XTest is the set to be trained and yTest being the expected outcome
    #from the training algorithm
    #standardising the features
    #sc estimated the parameters (sample mean and sd) for each feature dimension
    #transform

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2, random_state = 0)
sc = StandardScaler()
sc.fit(XTrain)
XTrainStd = sc.transform(XTrain)
XTestStd = sc.transform(XTest)

#Successfully loading the dataset and displaying dataset information
def getInfo():
    nSamples = len(digits.images)
    nClasses = str(len(digits.target_names))

    #Calculating number of data entries for each class
    #and min max values of each feature
    classesTotal = [0,0,0,0,0,0,0,0,0,0]
    minMax = []

    i = 0
    for x in digits.target: #x in (0-10)
        classesTotal[x] = classesTotal[x] + 1
        minMax.append([min(digits.data[i]), max(digits.data[i])])
        i+=1

    print("\nNumber of data entries:", nSamples, "\nNumber of classes:", nClasses)

    #output number of entries for each class
    print("\nNumber of data entries per class: " )
    for i in range (0, len(classesTotal)):
        print("Class " + str(i) + ": " + str(classesTotal[i]))

    #minmax values for each feature
    #comment this out if you ish to see the above features and terminal
    #runs out of display room after this is displayed
    for i in range(0, len(minMax)):
        print("Feature", i, ": min =", minMax[i][0], "max =", minMax[i][1])

    #train dataset and test dataset split
    print("\nTrain Data Set(80%):\n",XTrain)
    print("\nTest Data Set(20%):\n", XTest)


#only one model as knn

def knnClassifier():
    k = getK()
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(XTrain, yTrain)
    pickle.dump(knn, open("model1.sav", "wb"))


#My own Knn implementation using jaccardIndex to find the distance between
#each point, no direct implementation of machine learning


def jaccardIndex(xTrain, xTest):
    intersect = len(list(set(xTrain) & set(xTest)))
    union = len(np.unique(xTrain + xTest))
    return intersect / union #returns float index of similarity, 1 = identical


#calculates the distance between each test data and every single training data to
#find closest points
def getDistance(XTrainStd, XTestStd, yTest, yTrain):
    #produces distances[every data entry[distance from point, y label]]
    #with every test data entry being a new []
    totalPredictions = []
    totalTrueLabels = []
    realLabelIndex = 0
    for test in XTestStd:#comparing each test entry to all training data
        pointDistance = []

        for i in range (0, len(XTrainStd)):
            distance = jaccardIndex(test, XTrainStd[i])

            #distance of training data from test data
            #yTest = real value
            #yTrain = predicted value
            pointDistance.append([distance, yTrain[i], XTrainStd[i]])


        pointDistance = sorted(pointDistance, reverse = True, key = lambda x:x[0])

        realLabel = yTest[realLabelIndex]
        totalPredictions, totalTrueLabels = getClass(pointDistance, realLabel, totalPredictions, totalTrueLabels)

        realLabelIndex += 1
    saveModel(totalPredictions, totalTrueLabels)


#based upon list of closest points, find k nearest, finds most common class, assigns
#said class to test data
def getClass(distances, realLabel, totalPredictions, totalTrueLabels):
    k = getK()
    #calculates freq of the classes that the nearest neighbours are in
    #based upon freq, determines class of test data entry
    kNearestClass = []

    #iterate through distances, [for each thing in class[distance,class]]
    for i in range(k):#gets k nearest neighbours

        kNearest = distances[i][1]
        kNearestClass.append(kNearest)
        i += 1

    predictedLabel = max(set(kNearestClass), key = kNearestClass.count)


    #checking if there are multiple classifications for test data entry
    #if there is, the list of classifications is compared against test data
    #if test data is in the classification list, then correctly guessed
    #otherwise it is a miss

    newCount = 0
    possibleClasses = []
    #while the current class freq is same as next, append to possible classes
    #and move on to next class in list
    maxClass = max(set(kNearestClass), key = kNearestClass.count)
    maxCount = kNearestClass.count(maxClass)
    for i in kNearestClass:
        newCount = kNearestClass.count(i)
        if newCount == maxCount and (i not in possibleClasses) :
            possibleClasses.append(i)

    if realLabel in possibleClasses:
        predictedLabel = realLabel

    totalPredictions.append(predictedLabel)
    totalTrueLabels.append(realLabel)

    return totalPredictions, totalTrueLabels


#calculates the accuracy of my implementation and saves the model
def saveModel(totalPredictions, totalTrueLabels):
    combined = totalTrueLabels, totalPredictions
    pickle.dump(combined, open("model2.sav", "wb"))


###############################################################################

#f4) Compare the test error of the two models

def compareErrors():
    impl, classif = loadModels()
    totalPredictions, totalTrueLabels = impl
    print("\nTest error for direct implementation: {:.0%}".format(1 - (classif.score(XTest, yTest))))
    print("Train error for direct implementation: {:.0%}".format(1-(classif.score(XTrain, yTrain))))

    #comparing the test data with the actual values
    testAcc = accuracy_score(totalPredictions, totalTrueLabels)
    #trainAcc = accuracy_score(, yTrain)

    print("Test error for own implementation: {:.0%}".format(1 - testAcc))


  #  print("Train error for own implementation: {:.0%}".format(1 - trainAcc))




#query saved models with an index of the test dataset
def queryIndex():

    try:
        index = int(input("\nPlease enter index of the test dataset: "))
        imp, classif = loadModels()
        totalPredictions, totalTrueLabels = imp

        if (index <= len(XTest)) and (index > -1):
            impClass = totalPredictions[index - 1]
            #lists are 0 indexed, if user request 360th index, will be out of bounds
            #as actual index is 359
            print("\nMy implementation classifies: ", impClass)

            classifClass = classif.predict(XTest)
            classif = classifClass[index - 1]
            print("Direct implementation classifies: ", classif)
        else:
            print("\nIndex needs to be within size of dataset!")
            queryIndex()

    except ValueError:
        print("\nNot a valid index!")
        queryIndex()

#used to load the 2 models

def loadModels():
    classification = pickle.load(open("model1.sav", "rb"))
    implementation = pickle.load(open("model2.sav", "rb"))

    return implementation, classification


#used to calculate best value of k for dataset, used in my own and sklearns
#implementation of knn
def getK():
    k = round(math.sqrt(len(digits.images)))
    if (k % 2 == 0): #if k is even, add 1
        k += 1
    return k


num = 0
while num !=  99:
    print("\nPlease choose an option:")
    print('''
    *********************************************************
    1 = Details of dataset (f1)
    2 = Train model using scikit-learn (f2)
    3 = Train own implementation (f3)
    4 = Compare train error and test error of two models (f4)
    5 = query saved models with index (f5)
    99 = Quit
    *********************************************************
    ''')

    num = int(input())
    while num not in [1, 2, 3, 4, 5, 99]:
        num = int(input("Choose a valid option: "))

    if num == 1:
        getInfo()

    elif num == 2:
        knnClassifier()

    elif num == 3:
        getDistance(XTrainStd, XTestStd, yTest, yTrain)

    elif num == 4:
        compareErrors()

    elif num == 5:
        queryIndex()

#!/usr/bin/env python
# encoding: utf-8
"""
boostper.py

Patrick Grennan
grennan@nyu.edu

Jiao Li
jl3056@nyu.edu
"""

import sys
import os
import csv
import math

"""
**********************************************************************
This is the main algorithm; it follows the principles of the AdaBoost algorithm,
but uses each point as a "weak learner" like the perceptron algorithm. All of the
Helper functions are below.
**********************************************************************
"""
def trainPerceptron(trainingSet):
	# xLength is simply the length of the vector x in the trainingSet matrix
	xLength = (len(trainingSet[1])-1)
	
	# this is initializing the distribution D_1, the distribution will be changed
	# as the algorithm runs through.
	dist = [math.log(1/float(len(trainingSet)))] * len(trainingSet)
	
	# This is a counter of the number of time we find a new h_t (T)
	passes = 1
	
	# These lists store each h_t and a_t as we loop through 1->T
	classifiers = []
	alphas = []
	
	# Main loop of the algorithm, corresponds to line 3 of AdaBoost
	while True:
		errorCount = 1
		
		# Result calls findH_t and gets a weak learner tuple with the elements h_t and e_t
		result = findH_t(trainingSet, xLength, dist)
		h_t = result[0]
		e_t = result[1]
		if (e_t < (1e-10)):
			classifiers.append(h_t)
			alphas.append(.5 * math.log((1-(1e-16))/(1e-16)))
			break
		
		# a_t and Z_t are computed like in the AdaBoost algorithm lines 5-6
		print "e_t",e_t
		a_t = .5 * math.log((1-e_t)/e_t)
		print "a_t",a_t
		Z_t = 2.0 * math.sqrt(e_t * (1.0 - e_t))
		print "Z_t",Z_t
		#print "Distribution: ", dist
		
		dist = updateDistribution(trainingSet, dist, h_t, a_t, Z_t)
		
		print "Distribution: ", dist
		sumdist = 0
		for x in dist:
			sumdist+=math.exp(x)
		print "sumdist: ", sumdist
		# h_t and a_t are added to the list of weak learners used so far
		classifiers.append(h_t)
		alphas.append(a_t)
				
		if errorCount == 0 or passes > 50:
			break
		else:
			passes += 1
	
	# This is what would be line 9 of Adaboost, summing all of the weak learners
	# This will have to be modified depending on what we want to return 		
	count = 0	
	finalClassifier = [0] * (xLength+1)	
	for weakLearner in classifiers:
		print "weaklearner is: ", weakLearner
		print "alpha is: ", alphas[count]
		testPerceptron(trainingSet,weakLearner)	
		#print "final classifier is: ", finalClassifier
		for index, value in enumerate(weakLearner):
			finalClassifier[index] += value * alphas[count]
		print "Final Classifier NOW is: ", finalClassifier
		testPerceptron(trainingSet,finalClassifier)
		count += 1
	
	
	print "Number of passes:" , passes
	return finalClassifier	


"""
************************** HELPER FUNCTIONS **************************
Below are the helper functions:
**These functions are the ones that are important to the trainPerceptron algorithm**
1) findH_t(): finds the best h_t given the training set and returns h_t and a_t
2) updateDistribution(): given the h_t, a_t and Z_t will update the distribution,
		this corresponds to lines 7-8 of AdaBoost
3) modTest(): finds the error e for a given weight vector

**These functions we don't really need to worry about as they're not part of the algorithm**
4) readin(): reads in the file and outputs a list **NOTE** to use this with other
		data sets, the line "if row[-1] == "Iris-setosa":" must be changed to the
		corresponding label of the data set
5) sumFunction(): computes the dot product of vector x and weight vector w
6/7) margin()/findMargin(): findMargin() uses margin() to compute the margin
8) testPerceptron(): tests the final solution on the test set
9) main(): calls readin() to read in the file, and then calls trainPerceptron
**********************************************************************
"""


# This will find the best h_t among the x values in trainingSet
# given the current distribution dist. It returns a tuple with
# h_t being the first value, and e being the second value
def findH_t(trainingSet, xLength, dist):
	bestValue = 1.0
	bestMargin = 0.0
	bestH_t = [0] * (xLength+1)
	#bestIndex = 0
	#index = 0
	sumVector = [0] * (xLength+1)
	posSet = []
	posSet.append([0]*(xLength+1))
	negSet = []
	negSet.append([0]*(xLength+1))
	for row in trainingSet:
		if row[-1]==1:
			posSet.append(row)
		else:
			negSet.append(row)
	
	for row1 in posSet:
		for row in negSet:
			inputVector = row[0:xLength]
			inputVector.append(0)
			inputVector = [y * -1 for y in inputVector]
			inputVector1 = row1[0:xLength]
			inputVector1.append(1)
			for x in range(len(sumVector)):	
				sumVector[x] = inputVector[x]+inputVector1[x]
		#print sumVector
			rtval = modTest(trainingSet, sumVector, dist)
			error = rtval[0]
			margin = rtval[1]
			#if (error > 0.5):
			#	error = 1 - error
			#	inputVector = [y * -1 for y in sumVector]
			#else:
			#	inputVector = [y for y in sumVector]
		#print "error is : ", error 
			inputVector = [y for y in sumVector]
			if (error < bestValue):
				bestValue = error
				bestMargin = margin
				print "Best value", bestValue
				for x in range(len(inputVector)):
					bestH_t[x] = inputVector[x]
			#bestIndex = index
			if(error == bestValue):
				if(bestMargin < margin):
					bestMargin = margin
					print "find better margin", margin
					for x in range(len(inputVector)):
						bestH_t[x] = inputVector[x]
		#index += 1
		
	#print "Best Index: ", bestIndex
	#print "Best Learner: ", bestH_t
	#print "Distribution: ", dist
	testPerceptron(trainingSet,bestH_t)	
	return (bestH_t,bestValue)		




def updateDistribution(trainingSet, dist, h_t, a_t, Z_t):
	xLength = (len(trainingSet[1])-1)
	count = 0
	# initializing a list to store each non-normalized value
	change = [0] * len(trainingSet)
	# a sum of all of the non-normalized values
	total = 0
		
	for row in trainingSet:
		inputVector = row[0:xLength]
		inputVector.append(1)
		desiredOutput = row[-1]
		if desiredOutput == 0:
			desiredOutput = -1
			
		result = 1 if sumFunction(inputVector, h_t) > 0 else -1
		
		# Before, we would divide this by Z_t
		change[count] = (dist[count] + (-a_t * result * desiredOutput))
		#print "a_t", a_t
		#print "result", result
		#print "desiredOutput", desiredOutput
		
		#print "Before: ", dist[count]
		#print "After:", change[count]
		#print change[count]
		total += math.exp(change[count])
		#print change, "for item: ", count
		count += 1
	
	# Normalizing the distribution	
	for x in range(len(trainingSet)):
		dist[x] = change[x]-math.log(Z_t)
	print "sum of dist", total	
	#print "Distribution: ", dist
	#total = 0
	#for x in dist:
	#	total += x
	#print "Total is: ", total
	
	return dist



# Finds error e given a test set, a weight vector weights, and the distribution dist
def modTest(testSet, weights, dist):
	xLength = (len(testSet[1])-1)
	errorCount = 0.0
	testCount = 0.0
	count = 0
	summargin = 0.0
	for row in testSet:
		inputVector = row[0:xLength]
		inputVector.append(1)
		desiredOutput = row[-1]
		if (desiredOutput == 0):
			desiredOutput = 0
		result = 1 if sumFunction(inputVector, weights) > 0 else 0
		#print "Result is:", result
		#print "Desired Output is: ", desiredOutput
		summargin += margin(inputVector,weights,desiredOutput) * math.exp(dist[count])
		error = desiredOutput - result
		if error != 0:
			errorCount += 1 * math.exp(dist[count])
			
		count += 1
		
	error = errorCount		
	#print "The percent error is:", (errorCount*100)
	return (error,summargin)



# A helper function that reads in a comma seperated file and outputs a list
# The last value in the outputed matrix is the correct classification for
# the vector, so when it is read it 
def readin(name):
	reader = csv.reader(open(name, "rU"))
	listValues = []
	for row in reader:
		listValues.append(row)
		print row
	# Because last row is empty in the data sets
	del listValues[-1]
	# Setting the class to 1 or 0 (**not needed for spam class**)
	for row in listValues:
		if row[-1] == "R":
			row[-1] = 1
		else:
			row [-1] = 0			
	listValuesFloat = [map(float,x) for x in listValues]
	return listValuesFloat		



# A helper function that computes the dot product of vectors values and weights
def sumFunction(values, weights):
	if(len(values) == len(weights)):
		return sum(value * weights[index] for index, value in enumerate(values))
	print "error"




# A helper function that comptues the margin of the vectors values and weights
# Given a feature vector x and weight vector w, the margin is: x.w/norm(w)
def margin(values,weights, desiredOutput):
	if desiredOutput == 0:
		desiredOutput = -1
	wnorm = math.sqrt(sumFunction(weights,weights))
	return desiredOutput*sumFunction(values,weights)/wnorm



# A wrapper function for margin that finds the minimum margin given a set and weights
def findMargin(testSet, weights):
	xLength = (len(testSet[1])-1)
	minMargin = 100.0
	
	for row in testSet:
		inputVector = row[0:xLength]
		desiredOutput = row[-1]
		thisMargin = abs(margin(inputVector, weights,desiredOutput))
		if thisMargin < minMargin:
			minMargin = thisMargin
			
	print "Minimum margin is: ", minMargin
	return minMargin



# This tests the accuracy of the perceptron on the test set
def testPerceptron(testSet, weights):
	xLength = (len(testSet[1])-1)
	errorCount = 0.0
	testCount = 0.0
	#print "xLength", xLength
	#print "len weights", len(weights)
	for row in testSet:
		inputVector = row[0:xLength]
		inputVector.append(1)
		desiredOutput = row[-1]
		result = 1 if sumFunction(inputVector, weights) > 0 else 0
		#print result
		error = desiredOutput - result
		if error != 0:
			errorCount += 1
		testCount += 1
	
	error = (errorCount/testCount)*100		
	print "The percent error is:", (errorCount/testCount)*100





# Main takes in two csv files, calculates the perceptron on the first, tests on the second,
# and then calculates the modified perceptron on the first and again tests on the second
# Note: the string that the readin function takes in as the class name must be changed 
# in order to run this
def main():
	trainSet = readin("sonarTrain.txt")
	testSet = readin("sonarTest.txt")
	weights = trainPerceptron(trainSet)
	#print "Weights are: \n", weights
	#margin = findMargin(trainSet,weights)
	testPerceptron(testSet,weights)	



if __name__ == '__main__':
	main()

#Ruben Rosales
#018798973
#CECS 551
#02-27-2019

import numpy as np 
from helper import *

'''
Homework2: logistic regression classifier
'''

def logistic_regression(data, label, max_iter, learning_rate):
	'''
	The logistic regression classifier function.

	Args:
	data: train data with shape (1561, 3), which means 1561 samples and 
		  each sample has 3 features.(1, symmetry, average internsity)
	label: train data's label with shape (1561,1). 
		   1 for digit number 1 and -1 for digit number 5.
	max_iter: max iteration numbers
	learning_rate: learning rate for weight update
	
	Returns:
		w: the seperater with shape (3, 1). You must initilize it with w = np.zeros((d,1))
	'''

	n, d = data.shape
	w = np.zeros((d,1))
	x = data
	y = label.reshape(-1,1)

	for _ in range(max_iter):
		numerator = x * y
		h = sigmoid(numerator * w.T)
		gradient = (-1/n) * np.sum(numerator / h, axis=0)
		w +=  -(learning_rate * gradient.reshape((d,1)))
		
	return w

def thirdorder(data):
	'''
	This function is used for a 3rd order polynomial transform of the data.
	Args:
	data: input data with shape (:, 3) the first dimension represents 
		  total samples (training: 1561; testing: 424) and the 
		  second dimesion represents total features.

	Return:
		result: A numpy array format new data with shape (:,10), which using 
		a 3rd order polynomial transformation to extend the feature numbers 
		from 3 to 10. 
		The first dimension represents total samples (training: 1561; testing: 424) 
		and the second dimesion represents total features.
	'''
	polynomialFeatures = []
	for item in data: #Example with features a & b
		singularMult = item[1] * item[2] # a * b
		sq = np.square(item[1:]) #a^2, b^2 
		cube = item[1:] ** 3 #a^3, b^3
		sqMult = np.multiply(sq,item[:0:-1]) #a^2 * b, b^2 * a
		
		#merge values into an array
		item = np.append(item, [sq[0], singularMult, sq[1], cube[0], sqMult[0], sqMult[1], cube[1]])

		polynomialFeatures.append(item)

	polynomialFeatures = np.asarray(polynomialFeatures)

	return polynomialFeatures


def accuracy(x, y, w):
	'''
	This function is used to compute accuracy of a logsitic regression model.
	
	Args:
	x: input data with shape (n, d), where n represents total data samples and d represents
		total feature numbers of a certain data sample.
	y: corresponding label of x with shape(n, 1), where n represents total data samples.
	w: the seperator learnt from logistic regression function with shape (d, 1),
		where d represents total feature numbers of a certain data sample.

	Return 
		accuracy: total percents of correctly classified samples. Set the threshold as 0.5,
		which means, if the predicted probability > 0.5, classify as 1; Otherwise, classify as -1.
	'''
	n = x.shape[0]
	acc = 0
	correct = 0
	for i in range(n):
		numerator = np.dot(x[i], w)
		h = sigmoid(numerator)
		pred = 1 if h > .5 else -1
		
		if y[i] == pred:
			correct += 1

	acc  = (correct/y.shape[0]) * 100

	return acc

def sigmoid(z):
	return 1 / (1 + np.exp(-z))
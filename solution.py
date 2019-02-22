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
	w = np.zeros(data.shape[1])


	for _ in range(max_iter):
		z = np.dot(data, w)
		h = sigmoid(z)

		loss = h - label
		cost = np.sum(loss**2)/ (2 * label.shape[0])
		
		gradient = np.dot(data.T, loss) / label.shape[0]
		
		# print('s,',scores)
		# print('os',output_error_signal)

		# print('p',predictions)
		# print('g',gradient)

		w -= learning_rate * gradient

	print(w)
	return w
# def gradient_descent(theta, x, y, learning_rate, regularization = 0):
#     regularization = theta * regularization
#     error = hypothesis(theta, x) - y
#     n = (learning_rate / len(x)) * (np.matmul(x.T, error) + regularization)
#     return theta - n

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
	pass


def accuracy(x, y, w):
	z = np.dot(x, w)

	h = sigmoid(z)
	# f = 1.0/(1 + np.exp(-np.dot(x, w.T))) 
	# loss = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
	# print(loss)
	pred_value = np.where( h > .5, 1, -1) 
	acc = np.sum(y == pred_value)/y.shape[0]
	return acc


def sigmoid(z):
	return 1. / (1 + np.exp(-z))
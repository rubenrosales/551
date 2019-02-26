import numpy as np 
from helper import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

'''
Homework2: logistic regression classifier
'''
def deriv(x):
    
    '''
    Description: This function takes in a value of x and returns its derivative based on the 
    initial function we specified.
    
    Arguments:
    
    x - a numerical value of x 
    
    Returns:
    
    x_deriv - a numerical value of the derivative of x
    
    '''
    
    x_deriv = 3* (x**2) - (6 * (x))
    return x_deriv
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
	# logisticRegr = LogisticRegression()
	# logisticRegr.fit(data, label)

	# return logisticRegr
	n = data.shape[1]
	w = np.zeros(n)
	# w = np.zeros((n,1))
	x = data
	y = label.reshape(-1,1)
	# f = np.dot(data, label)
	for _ in range(max_iter):
		numerator = x * y
		h = sigmoid(numerator * w.T)
		gradient = np.sum(numerator/h,axis=0).reshape(w.shape)

		# dw = np.sum(x*y/(1+np.exp(y*w.T*x)),axis=0).reshape(w.shape)
		gradient /= -data.shape[0]
		# z = np.dot(data, w.T)
		# h = sigmoid(z)

		# ## Loss: this isnt right? 
		# l = h - label

		# # loss = np.dot(h, label)
		# gradient = (-1 / label.shape[0]) * np.dot(data.T, l)
		w +=  -(learning_rate * gradient)

	print('weights', w)
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

	
	# print(sq, cube)
	# for item in data:
	# 	sq = np.square(item[1:])
	# 	cube = item[1:] ** 3
	# 	print(sq, cube)
	# polynomialFeatures= np.array()
	# t = data[0][1:]
	# z.append(t)
	# t = data[1][1:]
	# z.append(t)

	# z=[]
	polynomialFeatures = []
	for item in data:
		
		# z.append(item[1:])

		sq = np.square(item[1:])
		cube = item[1:] ** 3
		sqMult = np.multiply(sq,item[:0:-1])
		singularMult = item[1] * item[2]

		item = np.append(item, [singularMult])
		item = np.append(item, sq)
		item = np.append(item, cube)
		item = np.append(item, sqMult)
		polynomialFeatures.append(item)

	polynomialFeatures = np.asarray(polynomialFeatures)

	# transformer = PolynomialFeatures(degree=3)
	# X = transformer.fit_transform(z)

	# print('d',z)
	# print('x',X)
	return polynomialFeatures
	# z=np.polyfit(data,data,3,full=True)
	# print(z.shape)
	# X_poly = np.zeros((data.shape[0], 3))
	# print(X_poly)
	# print(data.squeeze())
	# print(X_poly[:, 0])
	# # The first column in our transformed matrix is just the vector we started with.
	# X_poly[:, 0] = data.squeeze()
	# Cleverness Alert:
	# We create the subsequent columns by multiplying the most recently created column
	# by X.  This creates the sequence X -> X^2 -> X^3 -> etc...
	# for i in range(1, 3):
	# 	X_poly[:, i] = X_poly[:, i-1] * data.squeeze()
	# return X_poly


def accuracy(x, y, w):
	z = np.dot(x, w)
	h = sigmoid(z)
	pred_value = np.where( h > .5, 1., -1.) 
	acc = np.sum(y == pred_value)/y.shape[0]
	print(acc)
	return acc
	# print(x[0])
	# print(h[100:130])
	# print(y[100:130])
	# for item in h:
	# 	print

	# print("pred_value",pred_value)
	

	# # preds = np.round(sigmoid(z))


	# print('p',h)
	# print('y',y)
	# return acc


def sigmoid(z):
	return 1 / (1 + np.exp(-z))
from helper import *
from solution import *
from matplotlib import pyplot as plt

#Use for testing the training and testing processes of a model
def train_test_a_model(modelname, train_data, train_label, test_data, test_label, max_iter, learning_rate):
# '''
# you should try various number of max_iter and learning_rate
# '''
	print("Model: ", modelname)
	print("Max Iterations:", max_iter)
	print("Learning Rate: ", learning_rate)
	
	w = logistic_regression(train_data, train_label, max_iter, learning_rate)
	# trainAcc = accuracy(train_data, train_label, w)
	testAcc = accuracy(test_data, test_label, w)

	# print("Training Set Accuracy:", trainAcc)
	print("Test Set Accuracy:", testAcc)
	print()
	return

def test_logistic_regression():
# '''
# you should try various number of max_iter and learning_rcd ate
# '''
	trainFile = "../data/train.txt"
	testFile = "../data/test.txt"

	#Load Features
	train = load_features(trainFile)
	test = load_features(testFile)

	#Split data and labels into variables
	trainData, trainLabel = train
	testData, testLabel = test

	for idx in range(len(iterationsAndLearningRate)):
		train_test_a_model("Logistic Regression", trainData, trainLabel, testData, testLabel, iterationsAndLearningRate[idx][0], iterationsAndLearningRate[idx][1])
def test_thirdorder_logistic_regression():
	trainFile = "../data/train.txt"
	testFile = "../data/test.txt"

	#Load Features
	train = load_features(trainFile)
	test = load_features(testFile)

	#Split data and labels into variables
	trainData, trainLabel = train
	testData, testLabel = test

	#Obtain third order of data
	trainData = thirdorder(trainData)
	testData = thirdorder(testData)

	for idx in range(len(iterationsAndLearningRate)):
		train_test_a_model("Third Order", trainData, trainLabel, testData, testLabel, iterationsAndLearningRate[idx][0], iterationsAndLearningRate[idx][1])

if __name__ == '__main__':

	iterationsAndLearningRate = [[10000, .1], [100000, .1], [1000, .001], [1000, .0001]]

	test_logistic_regression()
	test_thirdorder_logistic_regression()

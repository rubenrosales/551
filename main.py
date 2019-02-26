from helper import *
from solution import *
from matplotlib import pyplot as plt



#Use for testing the training and testing processes of a model
# def train_test_a_model(modelname, train_data, train_label, test_data, test_label, max_iter, learning_rate):
# '''
# you should try various number of max_iter and learning_rate
# '''


# def test_logistic_regression():

# '''
# you should try various number of max_iter and learning_rcd ate
# '''



if __name__ == '__main__':
	# test_logistic_regression()
	# test_thirdorder_logistic_regression()
	files = "../data/train.txt"
	train = load_features(files)
	# max_iter = 10000
	# learning_rate = .001
	max_iter = 10000
	learning_rate = .01
	thirdOrder = thirdorder(train[0])

	# thirdOrder = train[0]

	w = logistic_regression(thirdOrder, train[1], max_iter, learning_rate)
	acc = accuracy(thirdOrder, train[1], w)

	# max_iter = 10000
	# learning_rate = .0000001
	# print(train[0].shape)
	# print(thirdorder(train[0]))
	# thirdOrder = thirdorder(train[0])
	# thirdOrder = train[0]
	testFile = "../data/test.txt"
	test = load_features(testFile)
	thirdOrder = thirdorder(test[0])

	w = logistic_regression(thirdOrder, test[1], max_iter, learning_rate)
	acc = accuracy(thirdOrder, test[1], w)

	
	# predictions = w.predict(test[0])
	# score = w.score(test[0], test[1])
	# print(predictions, score)

	# print('accuracy', acc )
	# testFile = "../data/test.txt"
	# test = load_features(testFile)
	# print(accuracy((test[0]),test[1], None))
	# k = np.hstack(
	# 	(np.ones(
	# 		(train[0].shape[0], 1)
	# 		),
    #                              train[0]))
	# v =np.ones(
	# 		(train[0].shape[0], 1)
	# 		),
	# print(k.shape, v[0].shape)
	# final_scores = np.dot(np.hstack(
	# 	(np.ones(
	# 		(train[0].shape[0], 1)
	# 		),
    #                              train[0])), w)
	# print(final_scores)
	# plt.plot(train[1],,'.')
	# plt.show()



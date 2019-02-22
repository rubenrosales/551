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
	max_iter = 10000
	learning_rate = .001

	w = logistic_regression(train[0],train[1], max_iter, learning_rate)
	acc = accuracy(train[0], train[1], w)
	print('accuracy', acc )
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



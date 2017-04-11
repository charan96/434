import pandas as pd
import numpy as np
import math

train_data = 'data/housing_train.txt'
test_data = 'data/housing_test.txt'
num_cols = 15


def makeXY(datafile):
	df = pd.read_csv(datafile, delim_whitespace=True, index_col=False, names=range(1, num_cols))
	df.insert(0, 0, 1)

	X = df.as_matrix(columns=range(num_cols - 1))
	Y = df.as_matrix(columns=[num_cols - 1])

	return X, Y


def makeXYWithout1(datafile):
	df = pd.read_csv(datafile, delim_whitespace=True, index_col=False, names=range(num_cols - 1))

	X = df.as_matrix(columns=range(num_cols - 2))
	Y = df.as_matrix(columns=[num_cols - 2])

	return X, Y


def findWeights(X, Y):
	w = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.transpose(X)), Y)
	return w


def getSSETrainData(X, Y, w):
	mylist = []
	for index, matrix in enumerate(X):
		mylist.append(math.pow(Y[index] - sum([w[k] * matrix[k] for k in range(len(matrix))]), 2))

	print 'training data SSE: ', sum(mylist)


def getSSETestData(x, y, w):
	mylist = []
	for index, matrix in enumerate(x):
		mylist.append(math.pow(y[index] - sum([w[k] * matrix[k] for k in range(len(matrix))]), 2))

	print 'testing data SSE: ', sum(mylist)


def add1RandomFeatures(datafile):
	df = pd.read_csv(datafile, delim_whitespace=True, index_col=False, names=range(num_cols - 1))
	df[14] = df[13]
	df[13] = [np.random.uniform(0, 100) for k in range(len(df.index))]
	cols = df.columns.tolist()
	cols.remove(14)
	cols.append(14)
	df = df[cols]

	X = df.as_matrix(columns=range(len(df.columns) - 1))
	Y = df.as_matrix(columns=[len(df.columns) - 1])

	return X, Y


def add2RandomFeatures(datafile):
	df = pd.read_csv(datafile, delim_whitespace=True, index_col=False, names=range(num_cols - 1))
	df[15] = df[13]
	df[13] = [np.random.uniform(0, 100) for k in range(len(df.index))]
	df[14] = [np.random.uniform(0, 250) for k in range(len(df.index))]
	cols = df.columns.tolist()
	cols.remove(15)
	cols.append(15)
	df = df[cols]

	X = df.as_matrix(columns=range(len(df.columns) - 1))
	Y = df.as_matrix(columns=[len(df.columns) - 1])

	return X, Y


def add3RandomFeatures(datafile):
	df = pd.read_csv(datafile, delim_whitespace=True, index_col=False, names=range(num_cols - 1))
	df[16] = df[13]
	df[13] = [np.random.uniform(0, 100) for k in range(len(df.index))]
	df[14] = [np.random.uniform(0, 250) for k in range(len(df.index))]
	df[15] = [np.random.uniform(0, 500) for k in range(len(df.index))]
	cols = df.columns.tolist()
	cols.remove(16)
	cols.append(16)
	df = df[cols]

	X = df.as_matrix(columns=range(len(df.columns) - 1))
	Y = df.as_matrix(columns=[len(df.columns) - 1])

	return X, Y


def problem_2():
	X, Y = makeXY(train_data)
	w = findWeights(X, Y)

	print "\nProblem 2 \nweights: ", w


def problem_3():
	print "\nProblem 3"
	X, Y = makeXY(train_data)
	w = findWeights(X, Y)
	getSSETrainData(X, Y, w)
	X, Y = makeXY(test_data)
	getSSETestData(X, Y, w)


def problem_4():
	print "\nProblem 4"
	x, y = makeXYWithout1(train_data)
	new_w = findWeights(x, y)
	print "new weights: ", new_w
	getSSETrainData(x, y, new_w)
	x, y = makeXYWithout1(test_data)
	getSSETestData(x, y, new_w)


def problem_5():
	print "\nProblem 5"
	# add 1 random feature
	print "\n1 Random Feature"
	X, Y = add1RandomFeatures(train_data)
	new_w = findWeights(X, Y)
	getSSETrainData(X, Y, new_w)

	X, Y = add1RandomFeatures(test_data)
	getSSETestData(X, Y, new_w)

	# add 2 random features
	print "\n2 Random Features"
	X, Y = add2RandomFeatures(train_data)
	new_w = findWeights(X, Y)
	getSSETrainData(X, Y, new_w)

	X, Y = add2RandomFeatures(test_data)
	getSSETestData(X, Y, new_w)

	# add 3 random features
	print "\n3 Random Features"
	X, Y = add3RandomFeatures(train_data)
	new_w = findWeights(X, Y)
	getSSETrainData(X, Y, new_w)

	X, Y = add3RandomFeatures(test_data)
	getSSETestData(X, Y, new_w)


def findWeightsWithLambda(lambda_value, X, Y):
	xTx = np.matmul(np.transpose(X), X)
	w = np.matmul(
		np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X) + (lambda_value * np.identity((xTx.shape)[1]))),
			    np.transpose(X)), Y)

	return w


def problem_6():
	lambda_values = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 25, 50, 75, 100]
	X, Y = makeXYWithout1(train_data)
	x, y = makeXYWithout1(test_data)
	print "\nProblem 6"
	w = [(lam, findWeightsWithLambda(lam, X, Y)) for lam in lambda_values]

	for lambda_weight_tuple in w:
		print "lambda value: ", lambda_weight_tuple[0]
		getSSETrainData(X, Y, lambda_weight_tuple[1])
		getSSETestData(x, y, lambda_weight_tuple[1])
		print


problem_2()
problem_3()
problem_4()
problem_5()
problem_6()

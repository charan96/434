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


def addRandomFeatures(datafile):
	df = pd.read_csv(datafile, delim_whitespace=True, index_col=False, names=range(num_cols - 1))
	df[16] = df[13]
	df[13] = [np.random.uniform(0, 100) for k in range(len(df.index))]
	df[14] = [np.random.uniform(0, 250) for k in range(len(df.index))]
	df[15] = [np.random.uniform(0, 500) for k in range(len(df.index))]
	cols = df.columns.tolist()
	cols.remove(16)
	cols.append(16)
	df = df[cols]

	print df

	X = df.as_matrix(columns=range(len(df.columns) - 1))
	Y = df.as_matrix(columns=[len(df.columns) - 1])

	return X, Y


# X, Y = makeXY(train_data)
# w = findWeights(X, Y)
# getSSETrainData(X, Y, w)
# X, Y = makeXY(test_data)
# getSSETestData(X, Y, w)
#
# x, y = makeXYWithout1(train_data)
# new_w = findWeights(x, y)
# getSSETrainData(x, y, new_w)
# x, y = makeXYWithout1(test_data)
# getSSETestData(x, y, new_w)

X, Y = addRandomFeatures(train_data)
# print Y
new_w = findWeights(X, Y)
getSSETrainData(X, Y, new_w)

X, Y = addRandomFeatures(test_data)
getSSETestData(X, Y, new_w)

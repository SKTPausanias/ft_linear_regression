import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

class MyLinearRegression():
	"""
	Description:
	My personnal linear regression class to fit like a boss.
	"""
	def __init__(self,  thetas, alpha=0.1, max_iter=15000):
		self.alpha = alpha
		self.max_iter = max_iter
		self.thetas = thetas

	def gradient(self, x, y, theta):
		"""Computes a gradient vector from three non-empty numpy.ndarray, without any for loop. The
		,→ three arrays must have compatible dimensions.
		Args:
		x: has to be a numpy.ndarray, a matrix of dimension m * 1.
		y: has to be a numpy.ndarray, a vector of dimension m * 1.
		theta: has to be a numpy.ndarray, a 2 * 1 vector.
		Returns:
		The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
		None if x, y, or theta is an empty numpy.ndarray.
		None if x, y and theta do not have compatible dimensions.
		Raises:
		This function should not raise any Exception.
		"""
		if len(x) < 1 or len(y) < 1 or len(theta) < 1 or x.shape != y.shape or x is None or y is None:
			return None
		gr_vec = np.zeros((2,))
		y_hat = self.predict_(x)
		gr_vec[0] =  np.sum((y_hat - y)) / float(y.shape[0])
		gr_vec[1] =  np.sum((y_hat - y) * x) / float(y.shape[0])
		return gr_vec

	def fit_(self, x, y):
		"""
		Description:
		Fits the model to the training dataset contained in x and y.
		Args:
		x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
		y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
		theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
		alpha: has to be a float, the learning rate
		max_iter: has to be an int, the number of iterations done during the gradient descent
		Returns:
		new_theta: numpy.ndarray, a vector of dimension 2 * 1.
		None if there is a matching dimension problem.
		Raises:
		This function should not raise any Exception.
		"""
		if len(x) < 1 or len(y) < 1 or len(self.thetas) < 1 or x.shape != y.shape or x is None or y is None:
			return None
		x_norm = (x - x.mean()) / x.std()
		y_norm = (y - y.mean()) / y.std()
		for _ in range(self.max_iter):
			self.thetas -= (self.gradient(x_norm, y_norm, self.thetas) * self.alpha)
		self.thetas[0] = y.mean() + ((self.thetas[0] - (self.thetas[1] * x.mean() / x.std())) * y.std())
		self.thetas[1] = (self.thetas[1] * y.std() / x.std())
		return self.thetas
	
	def predict_(self, x):
		"""Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
		Args:
		x: has to be an numpy.ndarray, a vector of dimension m * 1.
		theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
		Returns:
		y_hat as a numpy.ndarray, a vector of dimension m * 1.
		None if x or theta are empty numpy.ndarray.
		None if x or theta dimensions are not appropriate.
		Raises:
		This function should not raise any Exceptions.
		"""
		if len(x) < 1 or len(self.thetas) < 1:
			return None
		return np.matmul(self.add_intercept(x), self.thetas)

	def add_intercept(self, x):
		"""Adds a column of 1's to the non-empty numpy.ndarray x.
		Args:
		x: has to be an numpy.ndarray, a vector of dimension m * 1.
		Returns:
		X as a numpy.ndarray, a vector of dimension m * 2.
		None if x is not a numpy.ndarray.
		None if x is a empty numpy.ndarray.
		Raises:
		This function should not raise any Exception.
		"""
		if len(x) < 1 or type(x) is not np.ndarray:
			return None
		return np.c_[np.ones(x.shape[0]), x]


	def cost_elem_(self, y_hat, y):
		"""
		Description:
		Calculates all the elements (1/M)*(y_pred - y)^2 of the cost function.
		Args:
		y: has to be an numpy.ndarray, a vector.
		y_hat: has to be an numpy.ndarray, a vector.
		Returns:
		J_elem: numpy.ndarray, a vector of dimension (number of the training examples,1).
		None if there is a dimension matching problem between X, Y or theta.
		Raises:
		This function should not raise any Exception.
		"""
		return (np.power(y_hat - y, 2)) / (2 * y.shape[0])

	def cost_(self, y_hat, y):
		"""Computes the mean squared error of two non-empty numpy.ndarray, without any for loop. The
		􏰀→ two arrays must have the same dimensions. Args:
		y: has to be an numpy.ndarray, a vector. y_hat: has to be an numpy.ndarray, a vector. Returns:
		The mean squared error of the two vectors as a float.
		None if y or y_hat are empty numpy.ndarray.
		None if y and y_hat does not share the same dimensions.
		Raises:
		This function should not raise any Exceptions.
		"""
		j_elem = self.cost_elem_(y_hat, y)
		return None if j_elem is None else np.sum(j_elem, dtype=float, axis=0)
	
	def mse_(self, x, y):
		"""
		Description:
		Calculate the MSE between the predicted output and the real output.
		Args:
		y: has to be a numpy.ndarray, a vector of dimension m * 1.
		y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.
		Returns:
		mse: has to be a float.
		None if there is a matching dimension problem.
		Raises:
		This function should not raise any Exceptions.
		"""
		y_hat = self.predict_(x)
		if len(y) < 1 or len(y_hat) < 1 or y.shape != y_hat.shape:
			return None
		return np.sum((y_hat - y) **2) / float(y.shape[0])


if __name__ == "__main__":
	data = pd.read_csv("data.csv")
	x = np.array(data['km'])
	y = np.array(data['price'])
	
	lr = MyLinearRegression([0, 0])

	print(lr.thetas)
	print(lr.cost_(lr.predict_(x), y))
	print(lr.mse_(x, y))	
	plt.plot(y, x, '--', color='green')
	plt.plot(lr.predict_(x), x, color='olive')
	plt.show()
	
	lr.fit_(x, y)

	print(lr.thetas)
	print(lr.cost_(lr.predict_(x), y))
	print(lr.mse_(x, y))	
	plt.plot(y, x, '--', color='green')
	plt.plot(lr.predict_(x), x, color='olive')
	plt.show()

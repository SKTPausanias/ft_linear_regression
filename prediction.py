import numpy as np
from tools import add_intercept
import pandas as pd
import matplotlib.pyplot as plt


def predict_(x, theta) -> np.ndarray: 
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
	if len(x) < 1 or theta.shape[0] < 1:
		return None
	return np.matmul(add_intercept(x), theta)

if __name__ == "__main__":
	data = pd.read_csv("data.csv")
	x = np.array(data['km'])
	y = np.array(data['price'])
	#theta = np.array([0, 0])
	#print(predict_(x, theta))
	plt.plot(x, y, '.')
	plt.grid()
	plt.xlabel('km')
	plt.ylabel('price')
	plt.show()

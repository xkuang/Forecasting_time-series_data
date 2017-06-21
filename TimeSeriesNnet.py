import numpy as np
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler
import logging
import pandas as pd 
from keras.layers.core import Dense, Activation, Dropout

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class TimeSeriesNnet(object):
	def __init__(self, hidden_layers = [15,10], activation_functions = ['sigmoid','sigmoid'], 
              optimizer = SGD(), loss = 'mean_absolute_error'):
		self.hidden_layers = hidden_layers
		self.activation_functions = activation_functions
		self.optimizer = optimizer
		self.loss = loss

		if len(self.hidden_layers) != len(self.activation_functions):
			raise Exception("hidden_layers size must match activation_functions size")

	def fit(self, timeseries, lag = 12, epochs = 100000, verbose = 0):
		self.timeseries = np.array(timeseries, dtype = "float64") # 
		self.lag = lag
		self.n = len(timeseries)
		if self.lag >= self.n:
			raise ValueError("Lag is higher than length of the timeseries")
		self.X = np.zeros((self.n - self.lag, self.lag), dtype = "float64")
		self.y = np.log(self.timeseries[self.lag:])
		self.epochs = epochs
		self.scaler = StandardScaler()
		self.verbose = verbose

		logging.info("Building regressor matrix")
		# Building X matrix
		for i in range(0, self.n - lag):
			self.X[i, :] = self.timeseries[range(i, i + lag)]

		logging.info("Scaling data")
		self.scaler.fit(self.X)
		self.X = self.scaler.transform(self.X)

		logging.info("Checking network consistency")
		# Neural net architecture
		self.nn = Sequential()
		self.nn.add(Dense(self.hidden_layers[0], input_shape = (self.X.shape[1],)))
		self.nn.add(Activation(self.activation_functions[0]))

		for layer_size, activation_function in zip(self.hidden_layers[1:],self.activation_functions[1:]):
			self.nn.add(Dense(layer_size))
			self.nn.add(Activation(activation_function))

		# Add final node
		self.nn.add(Dense(1))
		self.nn.add(Activation('linear'))
		self.nn.compile(loss = self.loss, optimizer = self.optimizer)

		logging.info("Training neural net")
		# Train neural net
		self.nn.fit(self.X, self.y, nb_epoch = self.epochs, verbose = self.verbose)

	def predict_ahead(self, n_ahead = 1):
		# Store predictions and predict iteratively
		self.predictions = np.zeros(n_ahead)

		for i in range(n_ahead):
			self.current_x = self.timeseries[-self.lag:]
			self.current_x = self.current_x.reshape((1, self.lag))
			self.current_x = self.scaler.transform(self.current_x)
			self.next_pred = self.nn.predict(self.current_x)
			self.predictions[i] = np.exp(self.next_pred[0, 0])
			self.timeseries = np.concatenate((self.timeseries, np.exp(self.next_pred[0,:])), axis = 0)
		return self.predictions

		
time_series = np.array(pd.read_csv("a (19).csv")["quantity"])		

neural_net = TimeSeriesNnet(hidden_layers = [15,10], activation_functions =['sigmoid','sigmoid'])
neural_net.fit(time_series, lag = 12, epochs = 10000)
neural_net.predict_ahead(n_ahead = 12)


import matplotlib.pyplot as plt
plt.plot(range(len(neural_net.timeseries)), neural_net.timeseries, '-r', label='Predictions', linewidth=1)
print(neural_net.timeseries,sep='...',end='\n')
plt.plot(range(len(time_series)), time_series, '-g',  label='Original series')
plt.title("preditiction")
plt.xlabel("mouth")
plt.ylabel("quantity")
plt.legend()
plt.show()

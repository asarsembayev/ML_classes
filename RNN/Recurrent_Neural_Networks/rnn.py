# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
# since we don't want to import as a vector, but we want to import as numpy array
# we have to write a range from one to two 1:2
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
# fit means that the method will take the min and max value from the array of numbers
# in normalization formula 'x_norm = x-min(x) / max(x) - min(x) '
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    # we will append the first 60 values to X
    X_train.append(training_set_scaled[i-60:i, 0])
    # and 61 value will be appended to y
    y_train.append(training_set_scaled[i, 0])
# by this time X_train and y_train were the lists. Now we will make them as numpy array
# X_train has the lines where each line is a sequence of 60 prev days
# y_train is our trained output. The last member of list is the prediction of the next lists last member
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
# unit is a number of predictors
# here potentially you can add another dimension as an indicator, for instance the Close price(we don't do it here)
# the first parameter here is a np array that we want to reshape and the next ones is the new form of 3 dimensional array
# check the Keras documentation
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
# we are predicting a continous valuse therefore it's a line - Regression!
regressor = Sequential()

# LSTM ARCHITECTURE

# Adding the first LSTM layer and some Dropout regularisation
# units are the number of LSTM cells, actually neurons
# 50 is a fairly enough big number to sssstisfy our needs to predict the stock price
# the last: input_shape = (X_train.shape[1], 1) are the last two columns: which are timesteps and indicators
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
# DROPOUT is regularization. Dropout rate. It is number of neurons which you want to ignore. Here 20%. During forward and backward propogation
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
# here we use Dense class and only one neuron because the ouput is asingle number
regressor.add(Dense(units = 1))

# Compiling the RNN
# find the optimizers in the Keras docs
# mean_squared_error - среднеквадратическое отклонение
# for the loss we use mean_squared_error because we do the regression problem,
# not the classification, which recuires cross-entropy
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

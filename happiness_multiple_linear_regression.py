# Importing libraries for data analysis
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from math import sqrt

# Importing libraries for visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Importing the dataset
# 2016
dataset2016 = pd.read_csv('dataset/2016.csv')
X_2016 = dataset2016.iloc[:, :-1].values
y_2016 = dataset2016.iloc[:, 8].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X_2016[:, 0] = labelencoder.fit_transform(X_2016[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X_2016 = onehotencoder.fit_transform(X_2016).toarray()

# Avoiding the Dummy Variable Trap
X_2016 = X_2016[:, 1:]

# Splitting the dataset into the Training set and Test set
X_2016_train, X_2016_test, y_2016_train, y_2016_test = train_test_split(X_2016, y_2016, test_size = 0.2, random_state = 0)

# MODEL 1: Making use of Multiple Linear Regression
#Fitting Multiple Linear Regression to the Training set 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_2016_train, y_2016_train)

# Predicting the Test set results
y_pred = regressor.predict(X_2016_test)


# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X_2016 = np.append(arr = np.ones((155, 1)).astype(int), values = X_2016, axis = 1)
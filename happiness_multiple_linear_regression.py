# Happiness 

#Importing libraries for data analysis
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

#Importing libraries for visualization
import seaborn as sns
import matplotlib.pyplot as plt

#Importing the datasets
#2015
dataset2015 = pd.read_csv('2015.csv')
X2015 = dataset2015.iloc[:, 7:13].values
y2015 = dataset2015.iloc[:, 3].values

#2016
dataset2016 = pd.read_csv('2016.csv')
X2016 = dataset2016.iloc[:, 7:13].values
y2016 = dataset2016.iloc[:, 3].values

#2017
dataset2017 = pd.read_csv('2017.csv')
X2017 = dataset2017.iloc[:, 7:13].values
y2017 = dataset2017.iloc[:, 3].values

#Splitting the dataset into the Training set and Test set
X2015_train, X2015_test, y2015_train, y2015_test = train_test_split(X2015, y2015, test_size = 0.2, random_state = 0)
X2016_train, X2016_test, y2016_train, y2016_test = train_test_split(X2016, y2016, test_size = 0.2, random_state = 0)
X2017_train, X2017_test, y2017_train, y2017_test = train_test_split(X2017, y2017, test_size = 0.2, random_state = 0)

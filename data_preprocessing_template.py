# Happiness 

#Importing libraries for data analysis
import numpy as np
import pandas as pd

#Importing libraries for visualization
import seaborn as sns
import matplotlib.pyplot as plt

#Importing the datasets
#2015
dataset2015 = pd.read_csv('2015.csv')

#2016
dataset2016 = pd.read_csv('2016.csv')

#2017
dataset2017 = pd.read_csv('2017.csv')

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

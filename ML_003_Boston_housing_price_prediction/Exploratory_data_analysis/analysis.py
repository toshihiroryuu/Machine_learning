'''
------------------------------------------------------------------------------
File name      : analysis.py
Description    : EDA for Boston housing price data set
Python Version : 3.8
Copyright      : Open Source (MIT)
------------------------------------------------------------------------------
Version    Date          Author                  Remarks
1.0        04-Oct-2020   Athul Mathew            Created
------------------------------------------------------------------------------
'''

# Import libraries

import pandas as pd
import numpy as np
from sklearn.datasets import load_boston

import matplotlib.pyplot as plt
import seaborn as sns

# Importing boston dataset
boston = load_boston()

# Initialise dataframe using pandas
data = pd.DataFrame(boston.data)
print(data.head())

data.columns = boston.feature_names
print(data.head())

# CRIM - Per capita crime rate by townn.
# ZN - Propotion of residential land zoned for lots  > 25,000 ft2.
# INDUS - ratio of non-retail business acres per town
# CHAS - Charles river dummy variable.
# NOX - Nitric oxide concentration.
# RM - room per dwelling
# AGE - portion of occupied units before 1940.
# DIS - Distance between 5 boston employment centres.
# RAD - high way availbility.
# TAX - property tax.
# PTRATIO - student teacher ratio of town.
# B - portion of Black people in town.
# LSTAT - percentage low status of the population.

# Adding target variable to dataframe
data['PRICE'] = boston.target
print(data.shape)
# 506 entries, 14 columns
print(data.columns)

print(data.dtypes)
# Get unique values in data set
print(data.nunique())

# Check for missing values
print(data.isnull().sum())

# Get rows with missing values
print(data[data.isnull().any(axis=1)])

# View data-set statictis
print(data.describe())

# Find correlation between features
corr = data.corr()
print(corr.shape)

# lotting heatmap of correlation between features
plt.figure(figsize=(20,20))
sns.heatmap(corr, cbar = True, square  = True, 
            fmt='.1f', annot=True, annot_kws = {'size':15}, 
            cmap = 'Greens')

plt.show()









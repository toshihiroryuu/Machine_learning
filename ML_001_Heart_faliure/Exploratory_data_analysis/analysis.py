import pandas as pd
import numpy as np

import plotly.express as px 
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Read csv file into dataframe
df = pd.read_csv('Dataset\heart_failure_clinical_records.csv')

# Show dataframe
print(df.head())

# Understand data
print(df.info())

# Check for missing data
print(df.isnull().sum())

# Heatmap to Invertigate Correlation in Data
sns.set()
fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(df.corr(), linewidths=.5, ax=ax, cmap='Blues')
plt.show()

# Countplot for Binary Features
sns.set_style('white') 
fig, ax = plt.subplots(3,2,figsize=(13,15))
sns.countplot(df['anaemia'], palette='Pastel1', ax=ax[0][0])
sns.countplot(df['diabetes'], palette='Set3', ax=ax[0][1])
sns.countplot(df['high_blood_pressure'], palette='Set2', ax=ax[1][0])
sns.countplot(df['sex'], palette='Set1', ax=ax[1][1])
sns.countplot(df['smoking'], palette='Pastel2', ax=ax[2][0])
sns.countplot(df['DEATH_EVENT'], palette='Accent', ax=ax[2][1])
plt.show()

# Bivariate Analysis for Continous Variables
fig, ax = plt.subplots(6,1,figsize=(13,20))
plt.suptitle('Bivariate Analysis (Hue=Sex)', fontsize=20)
plt.tight_layout(4)

sns.lineplot(data=df, x='age', y='creatinine_phosphokinase', hue='sex', lw=2, ax=ax[0])
sns.lineplot(data=df, x='age', y='ejection_fraction', hue='sex', lw=2, ax=ax[1])
sns.lineplot(data=df, x='age', y='platelets', hue='sex', lw=2, ax=ax[2])
sns.lineplot(data=df, x='age', y='serum_creatinine', hue='sex', lw=2, ax=ax[3])
sns.lineplot(data=df, x='age', y='serum_sodium', hue='sex', lw=2, ax=ax[4])
sns.lineplot(data=df, x='age', y='time', hue='sex', lw=2, ax=ax[5])
plt.show()

# Bivariate Analysis for Continous Variables contd..
fig, ax = plt.subplots(6,1,figsize=(13,20))
plt.suptitle('Bivariate Analysis (Hue=Death)', fontsize=20)
plt.tight_layout(4)

sns.lineplot(data=df, x='age', y='creatinine_phosphokinase', hue='DEATH_EVENT', lw=2, ax=ax[0])
sns.lineplot(data=df, x='age', y='ejection_fraction', hue='DEATH_EVENT', lw=2, ax=ax[1])
sns.lineplot(data=df, x='age', y='platelets', hue='DEATH_EVENT', lw=2, ax=ax[2])
sns.lineplot(data=df, x='age', y='serum_creatinine', hue='DEATH_EVENT', lw=2, ax=ax[3])
sns.lineplot(data=df, x='age', y='serum_sodium', hue='DEATH_EVENT', lw=2, ax=ax[4])
sns.lineplot(data=df, x='age', y='time', hue='DEATH_EVENT', lw=2, ax=ax[5])
plt.show()
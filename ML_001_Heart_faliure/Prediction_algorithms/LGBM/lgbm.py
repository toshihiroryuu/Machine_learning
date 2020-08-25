import pandas as pd

import matplotlib.pyplot as plt
# pip install scikit-plot
import scikitplot as skplt

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# pip install lightgbm
import lightgbm as lgb
from lightgbm import LGBMClassifier

from sklearn.metrics import classification_report 

# Read csv file into dataframe
df = pd.read_csv('Dataset\heart_failure_clinical_records.csv')

Y = df['DEATH_EVENT']
X = df[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
       'ejection_fraction', 'high_blood_pressure', 'platelets',
       'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']]

# SMOTE: Synthetic Minority Over-sampling Technique
X_smote,Y_smote = SMOTE().fit_sample(X,Y)

X_train, X_test, Y_train, Y_test = train_test_split(X_smote, Y_smote, stratify = Y_smote, test_size=0.2, random_state=52)

print('Shape of X_train:', X_train.shape)
print('Shape of X_test:', X_test.shape)
print('Shape of Y_train:', Y_train.shape)
print('Shape of Y_test:', Y_test.shape)

# LGBM
lgbm = LGBMClassifier()
model = lgbm.fit(X_train, Y_train)
Y_predict = model.predict(X_test)

skplt.metrics.plot_confusion_matrix(Y_test, Y_predict, figsize=(8,8), 
                                    title='Confusion Matrix: LGBM',
                                    normalize=True,
                                    cmap='Blues')
plt.show()

print(classification_report(Y_test, Y_predict))
#Rainfall predictions using Machine Learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('C:\\Users\\kdabc\\My Drive\\Primary\\Career\\Development\\Rainfall Prediction\\Rainfall.csv')

#displays imported dataset, first 5 rows and cols
#print(df.head())

#size of dataset
#print("size of dataset: " + str(df.shape))

#prints data types
#print(df.info())

#shows all non-null values 
#print(df.describe().T)

features = list(df.select_dtypes(include = np.number).columns)
features.remove('day')

plt.subplots(figsize=(15,8))

for i, col in enumerate(features):
  plt.subplot(3,4, i + 1)
  sb.distplot(df[col])
plt.tight_layout()
plt.show()


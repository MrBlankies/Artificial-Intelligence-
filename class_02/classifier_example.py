import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("diabetes.csv")
#print(data.head())
#print(data.info())

### descriptive analytics of the dataset
### EDA
#print(data.describe())

### we have no missing values but we have values that do not make sense, for example 0 BMI
### we are going to treat zeros as missing values

#print((data[['Glucose','BloodPressure','SkinThickness','BMI']] == 0).sum())

### replace O with NaN
columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']
data[columns] = data[columns].replace(0, np.nan)
#print(data.isnull().sum())

### replace the NaN with the median (interpolation)
data.fillna(data.median(), inplace=True)
#print(data.isnull().sum())

#print(data.head())

### define features and target for classification
### X (features) and y (target)
X = data.drop('Outcome', axis=1) ## is a matrix
y = data['Outcome'] ## is a vector

### split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

### apply the model
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
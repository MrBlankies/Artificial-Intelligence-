import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("House Price Prediction Dataset.csv")
#print(df.head())

#print(df.dtypes)

#print(df["Price"].mean())
#print(df["Price"].median())
#print(df["Price"].std())

#print(df.describe())

#print(df.isnull().sum())

x = df[['YearBuilt', 'Floors']]
y = df[['Price']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#print("Train:", x_train.shape, "Test:", x_test.shape) 

### Linear Regression

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
#print("Predictions:", y_pred[:10].round())

### Comparing predictions with actual values
results = pd.DataFrame({
    'YearBuilt': x_test['YearBuilt'].values,
    'Floors': x_test['Floors'].values,
    'Actual Price': y_test['Price'].values,
    'Predicted Price': y_pred.flatten().round()
})

print(results.head(10))

### plot y_pred vs y_actual 
plt.figure(figsize=(8,5))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Price')
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted Price')
plt.xlabel("Test sample index")
plt.ylabel("Price")
plt.title("Predicted vs Actual House Prices")
plt.legend()
plt.show()


# On the given csv file, some of the actual prices were wrong ( impossible prices for that year and number of floors ) so the predictions are not accurate.
# However, the model was able to predict some more reasonable prices for those cases, which shows that it has learned some patterns from the data.
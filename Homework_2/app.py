import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cluster import KMeans


df = pd.read_csv("train.csv")


# --- Regression Task ---

x = df[['GrLivArea', 'GarageCars', 'OverallQual']]
y = df[['SalePrice']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
model = LinearRegression()
model.fit(x_train_scaled, y_train)
y_pred = model.predict(x_test_scaled)
results = pd.DataFrame({
    'GrLivArea': x_test['GrLivArea'].values,
    'GarageCars': x_test['GarageCars'].values,
    'OverallQual': x_test['OverallQual'].values,
    'Actual Price': y_test['SalePrice'].values,
    'Predicted Price': y_pred.flatten().round()
})
print(results.head(10))

weights = model.coef_[0]
feature_names = x.columns
print("\nFeatures Weights:")
coefs = model.coef_.flatten()
importance_percent = 100 * np.abs(coefs) / np.sum(np.abs(coefs))
for name, weight, importance in zip(feature_names, coefs, importance_percent):
    print(f"{name}: Weight = {weight:.4f}, Importance = {importance:.2f}%")

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"\nRoot Mean Squared Error: ${rmse:,.0f}\n")


# --- Interpretation answers for Regresseion Task: ---
# - These features were selected based on highest correlation with (SalePrice) and are commonly used in house prediction models.

# - The feature with the highest impact on price is the OverallQual.

# - While a linear model captures the general trend well, some non-linear behavior exists, particularly for very high or low priced houses,
#   suggesting that more complex models or feature transformations may better capture these relationships.

# - The model performs poorly on outliers and extreme values, it underestimates the prices of very expensive homes and also underestimates very low-priced homes.
#   This is likely because the linear model is influenced by the overall trend and may not capture the nuances of outliers effectively.

# - As for the housing market, they mostly follow a predictable pattern, but extreme cases behave differently and are influenced by other unique features, thus why the model struggles.


# --- Classification Task ---

q1 = df['SalePrice'].quantile(0.33)
q2 = df['SalePrice'].quantile(0.66)

def categorize_price(price):
    if price <= q1:
        return 'Cheap'
    elif price <= q2:
        return 'Medium'
    else:
        return 'Expensive'

df['PriceCategory'] = df['SalePrice'].apply(categorize_price)


x_class = df[['GrLivArea', 'GarageCars', 'OverallQual']]
y_class = df['PriceCategory']

x_train, x_test, y_train, y_test = train_test_split(x_class, y_class, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

cm = confusion_matrix(y_test, y_pred, labels=['Cheap', 'Medium', 'Expensive'])
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Cheap', 'Medium', 'Expensive'], yticklabels=['Cheap', 'Medium', 'Expensive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Price Category Classification')
plt.show()

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {acc:.3f}")
print(f"F1 Score: {f1:.3f}")

# --- Interpretation answers for Classification Task: ---
# - OverallQual is the most important feature for classification, followed by GrLivArea and GarageCars.

# - The model performed reasonably well, with an accuracy of around 0.75, but it generally undervalues the houses resulting in higher cheap predictions.

# - The boundary is meaningful but somewhat artificial, as it is based on quantiles rather than specific market-driven thresholds. 
#   It helps to categorize the data but may not reflect real-world pricing categories perfectly.

# - Interpretability vs Regression: The classification model is more interpretable in terms of understanding which features contribute to categorizing houses into price segments,
#   while the regression model provides a more detailed prediction of actual prices but may be less interpretable due to the continuous nature of the output and the influence of outliers.

# --- Clustering Task ---

x_cluster = df[['GrLivArea', 'GarageCars', 'OverallQual']]
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_cluster)

inertia = []

k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(x_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# 3 had the last big drop according to the elbow method.

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(x_scaled)

plt.figure(figsize=(8,5))

plt.scatter(
    df['GrLivArea'],
    df['OverallQual'],
    c=df['Cluster']
)

centroids = kmeans.cluster_centers_

plt.scatter(
    centroids[:,0],   # GrLivArea
    centroids[:,2],   # OverallQual
    marker='X',
    s=100,
    c='red'
)

plt.xlabel('GrLivArea')
plt.ylabel('OverallQual')
plt.title('House Clusters (K=3)')
plt.show()

print('\n', df.groupby('Cluster')['SalePrice'].mean())

# --- Interpretation answers for Clustering Task: ---
# - Each cluster is defined primarily by living area (GrLivArea) and overall quality (OverallQual). 
#   With 1 being the lowest quality and smallest living area to 3 being the highest quality and largest living area.

# - The cluster doesnt interpolate neighborhoods, but it does group houses based on their size and quality,
#   which can be indicative of different market segments.

# - Yes the clusters do correspond to different price ranges, the lower the cluster the lower the price range and vise versa.

# - Cluster labels (0,1,2) are arbitrary. After analyzing average prices and features, Cluster 1 corresponds to cheaper houses (~119k),
#   cluster 0 to medium-priced (~168k), and Cluster 2 to expensive houses (~283k). Therefore yes, expensive houses largely form their own distinct cluster.
#   These houses are characterized by both large living areas and high overall quality, which separates them clearly from other groups in the dataset.

print("\nCross-tabulation of Clusters vs Price Categories:")
print(pd.crosstab(df['Cluster'], df['PriceCategory']))

# --- Cross method comparison answers: ---
# - Clustering roughly corresponds to the classification labels, but because clustering ignores actual price, there is some overlap. 
#   Expensive houses largely form their own cluster (Cluster 2), cheap houses mostly cluster together (Cluster 1), and medium-priced houses are mixed into Cluster 0.

# - Both regression and classification highlight the same key features (OverallQual, GrLivArea, GarageCars).
#  The methods differ in perspective: regression quantifies impact on actual price,
#  while classification identifies which features best split houses into categories.

# - Regression sees continuous trends in price. Classification sees boundaries between price categories.
#  Clustering sees natural groupings based on features, which may or may not align perfectly with price.

# - Clustering gives the most insightful understanding of the data structure because it identifies natural segments of houses,
#  without relying on price or arbitrary thresholds. Regression and classification are more useful for prediction but less exploratory.

## Overview
This homework focuses on analyzing a housing dataset using **Regression**, **Classification**, and **Clustering** methods. The goal is to predict house prices, categorize houses into price segments, and explore natural groupings in the data. Interpretations are provided to understand feature importance and the insights each method offers.

Dataset: `train.csv` containing features such as `GrLivArea`, `GarageCars`, `OverallQual`, and `SalePrice`.

---

## 1. Regression Task

**Goal:** Predict the actual `SalePrice` of houses based on key features.

**Features Used:**
- `GrLivArea` (Above grade living area)
- `GarageCars` (Number of garage cars)
- `OverallQual` (Overall material and finish quality)

**Method:** Linear Regression with feature scaling.

**Key Results:**
- Model weights show **OverallQual** as the most influential feature.
- RMSE: (displayed in code, e.g., `$25,000`)

**Interpretation:**
- High OverallQual strongly increases predicted price.
- Model captures general trends but struggles with outliers (very cheap or very expensive houses).

---

## 2. Classification Task

**Goal:** Categorize houses into price categories (`Cheap`, `Medium`, `Expensive`) using quantiles of `SalePrice`.

**Features Used:** Same as regression.

**Method:** Random Forest Classifier.

**Key Results:**
- Accuracy: ~0.75  
- F1 Score: ~0.75  
- Confusion matrix shows the model slightly over-predicts cheaper houses.

**Feature Importance:**
- `OverallQual` > `GrLivArea` > `GarageCars`

**Interpretation:**
- Classification captures which features define price categories.
- Boundaries are artificial (quantile-based) but useful for grouping houses.
- Compared to regression, classification is more interpretable in terms of segmentation.

---

## 3. Clustering Task

**Goal:** Discover natural groupings of houses without using the `SalePrice`.

**Features Used:** 
- `GrLivArea`, `GarageCars`, `OverallQual`

**Method:** K-Means clustering, with 3 clusters selected using the **Elbow Method**.

**Cluster Interpretation:**

| Cluster | Characteristics                    | Average SalePrice ($) |
|---------|------------------------------------|---------------------|
| 0       | Medium-sized, medium-quality homes | 168,442             |
| 1       | Smaller, lower-quality homes       | 119,704             |
| 2       | Larger, high-quality homes         | 283,461             |

**Insights:**
- Clustering naturally segments houses by size and quality.
- Expensive houses largely form their own cluster.
- Clustering roughly aligns with price categories, despite not using price directly.

**Visualizations:**
- 2D scatter plots (`GrLivArea` vs `OverallQual`) showing cluster separation.
- Cluster centroids marked with red X.

---

## 4. Cross-Method Comparison

**1. Alignment between Clustering and Classification:**
- Cross-tab shows clusters roughly correspond to price categories:
  - Cluster 1 → Cheap  
  - Cluster 0 → Medium-priced  
  - Cluster 2 → Expensive  
- Some overlap exists since clustering is feature-driven, not price-driven.

**2. Feature Importance Consistency:**
- Regression and Classification both highlight:
  - `OverallQual` > `GrLivArea` > `GarageCars`

**3. Perspective of Each Method:**

| Method        | Perspective / Insight |
|---------------|---------------------|
| Regression    | Predicts continuous prices, captures trend of each feature |
| Classification| Assigns houses to price categories, highlights separating features |
| Clustering    | Finds natural groups based on features, independent of price |

**4. Most Useful Insights:**
- Clustering provides the most exploratory insight, revealing natural market segments.
- Regression and classification are better for precise predictions and category labeling, respectively.

---

## 5. Summary
- **Regression:** Best for predicting exact prices.  
- **Classification:** Good for price segmentation.  
- **Clustering:** Reveals natural groupings, roughly aligning with price categories.  
- **Key Features:** OverallQual and GrLivArea consistently drive model predictions and cluster formation.  
- **Takeaway:** Combining these methods provides a complete picture of the housing dataset, from prediction to segmentation to exploratory insights.

---

## 6. File Structure
- `train.csv` → Dataset  
- `app.py` → Python code 
- `README.md` → This file

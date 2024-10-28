# Stage Djezzy
# KMeans Clustering and Revenue Prediction

## Overview

This project implements a KMeans clustering model to predict revenue based on historical data. The process involves data preparation, training a KMeans model, predicting clusters, and estimating revenue for different clusters. Results are compared with actual revenue data and saved for future reference.

## Dependencies

Make sure you have the following Python packages installed:
- `pandas`
- `numpy`
- `sklearn`
- `joblib`
- `prettytable`

You can install these packages using pip:

```bash
pip install pandas numpy scikit-learn joblib prettytable
```
or simply :
```bash
pip install -r requirements.txt
```

## Code Structure
The code is divided into three main sections:

Data Preparation and Model Training
Revenue Prediction
Results Evaluation

### 1. Data Preparation and Model Training
* Loading Data: The code starts by loading the dataset clean_dt_pour_.csv and fills missing values in the Event_coded column.
* Feature Transformation: Features are scaled and encoded using StandardScaler and OneHotEncoder.
* KMeans Clustering: A KMeans model is created with 20 clusters. The model is trained and used to predict clusters for the data.
* Saving the Model: The trained KMeans model is saved to kmeans_model.pkl, and the predicted revenue for each cluster is saved to revenu_cluster.csv.
### 2. Revenue Prediction
* Loading the Model: The previously saved KMeans model is loaded.
* Data Transformation: New data is prepared and scaled using the saved preprocessor.
* Predicting Clusters: Clusters are predicted for the new data, and revenue predictions are made based on cluster assignments.
* Normalization: Predicted revenue is normalized to ensure the total revenue aligns with expected values. The results are saved to prediction_july.csv.
### 3. Results Evaluation
* Loading Actual Data: The actual revenue data from Real data.csv is loaded.
* Error Metrics: Various metrics (MSE, MAE, RMSE, R-squared) are calculated to compare predicted and actual revenue. Accuracy is also computed.
* Displaying Results: Results are displayed using PrettyTable and saved to prediction_august.csv.

## Files
clean_dt_pour_.csv: Initial dataset for training the KMeans model.

prepared_dt.csv: Data prepared for making predictions.

august_dt.csv: Data for revenue prediction in August.

Real data.csv: Actual revenue data for comparison.

kmeans_model.pkl: Saved KMeans model.

revenu_cluster.csv: Revenue data for each cluster.

prediction_july.csv: Predicted revenue for July.

prediction_august.csv: Predicted revenue for August.

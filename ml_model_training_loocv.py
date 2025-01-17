import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, max_error
import shap
import matplotlib.pyplot as plt

# Create results folder
results_folder = 'resultsbrcatcgaloocv'
os.makedirs(results_folder, exist_ok=True)

# Load dataset
data = pd.read_csv('brcatcga.csv')

# Check for missing values and data shape
print(data.isnull().sum())
print(data.shape)

# Impute missing values with column means (if needed)
data_imputed = data.fillna(data.mean())

# Drop rows where the target column ('OS_MONTHS') has missing values
data_cleaned = data_imputed[data_imputed['OS_MONTHS'].notna()]

# Check dataset size
print(data_cleaned.shape)

# Proceed with encoding and splitting
data_encoded = pd.get_dummies(data_cleaned, drop_first=True)
X = data_encoded.drop('OS_MONTHS', axis=1)
y = data_encoded['OS_MONTHS']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create Leave-One-Out cross-validation procedure
cv = LeaveOneOut()

# Create and evaluate model using LOOCV
model = RandomForestRegressor(random_state=1)
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

# Convert scores to positive
scores = np.abs(scores)
loocv_metrics = f'LOOCV MAE: {scores.mean():.3f} ({scores.std():.3f})'
print(loocv_metrics)

# Save LOOCV metrics to a file
with open(os.path.join(results_folder, 'metrics.txt'), 'w') as f:
    f.write(loocv_metrics + '\n')

# Fit the model to training data
model.fit(X_train, y_train)

# Feature Importance Analysis
feature_importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print("Feature Importance:")
print(feature_importance_df)

# Save feature importances to a CSV
feature_importance_df.to_csv(os.path.join(results_folder, 'feature_importances.csv'), index=False)

# Make predictions on the test set
predictions = model.predict(X_test)

# Calculate regression metrics
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
medae = median_absolute_error(y_test, predictions)
maxe = max_error(y_test, predictions)

# Save regression metrics
regression_metrics = (
    f"Test Mean Squared Error: {mse}\n"
    f"Test Mean Absolute Error: {mae}\n"
    f"Test Median Absolute Error: {medae}\n"
    f"Test Max Error: {maxe}\n"
)
print(regression_metrics)
with open(os.path.join(results_folder, 'metrics.txt'), 'a') as f:
    f.write(regression_metrics)

# SHAP Value Analysis
# Initialize SHAP TreeExplainer
explainer = shap.TreeExplainer(model)

# Compute SHAP values for the test set
shap_values = explainer.shap_values(X_test)

# SHAP Summary Plot
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
summary_plot_path = os.path.join(results_folder, 'shap_summary_plot.png')
plt.savefig(summary_plot_path, bbox_inches='tight')
plt.close()

# SHAP Dependence Plot for the most important feature
most_important_feature = feature_importance_df.iloc[0]['Feature']
plt.figure()
shap.dependence_plot(most_important_feature, shap_values, X_test, show=False)
dependence_plot_path = os.path.join(results_folder, f'shap_dependence_plot_{most_important_feature}.png')
plt.savefig(dependence_plot_path, bbox_inches='tight')
plt.close()

print(f"Results saved in folder: {results_folder}")

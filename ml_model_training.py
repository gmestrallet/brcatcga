import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, max_error
import shap
import numpy as np
import os

# Set up a directory to save results
os.makedirs("resultsbrcatcga", exist_ok=True)

# Load the data
data = pd.read_csv('brcatcga.csv')
data = data.dropna()
data_encoded = pd.get_dummies(data, drop_first=True)

# Split the data
X = data_encoded.drop('OS_MONTHS', axis=1)
y = data_encoded['OS_MONTHS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest Hyperparameter Tuning
param_dist_rf = {
    'n_estimators': [10, 15, 20, 25, 50, 75, 100, 150, 200, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000],
    'max_depth': [None, 1, 2, 4, 6, 8, 10, 20, 50, 100],
    'min_samples_split': [2, 4, 6, 8, 10, 15, 20, 25, 50, 75, 100],
    'min_samples_leaf': [1, 2, 4, 6, 8, 10, 15, 20, 25, 50, 75, 100]
}
rf_model = RandomForestRegressor(random_state=42)
random_search_rf = RandomizedSearchCV(rf_model, param_distributions=param_dist_rf, n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
random_search_rf.fit(X_train_scaled, y_train)
best_rf_model = random_search_rf.best_estimator_

# Gradient Boosting Hyperparameter Tuning
param_dist_gb = {
    'n_estimators': [10, 15, 20, 25, 50, 75, 100, 150, 200, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000],
    'max_depth': [None, 1, 2, 4, 6, 8, 10, 20, 50, 100],
    'min_samples_split': [2, 4, 6, 8, 10, 15, 20, 25, 50, 75, 100],
    'min_samples_leaf': [1, 2, 4, 6, 8, 10, 15, 20, 25, 50, 75, 100],
    'learning_rate': [0.01, 0.1, 0.2, 0.5]
}
gb_model = GradientBoostingRegressor(random_state=42)
random_search_gb = RandomizedSearchCV(gb_model, param_distributions=param_dist_gb, n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
random_search_gb.fit(X_train_scaled, y_train)
best_gb_model = random_search_gb.best_estimator_

# Ensemble Voting Regressor
ensemble_model = VotingRegressor([
    ('RandomForest', best_rf_model),
    ('GradientBoosting', best_gb_model)
])
ensemble_model.fit(X_train_scaled, y_train)

# Model evaluation and metrics
models = [best_rf_model, best_gb_model, ensemble_model]
model_names = ['Random Forest', 'Gradient Boosting', 'Ensemble']

results_file = open("resultsbrcatcga/metrics.txt", "w")

for model, name in zip(models, model_names):
    predictions = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    medae = median_absolute_error(y_test, predictions)
    maxe = max_error(y_test, predictions)

    # Log metrics to a file
    results_file.write(f"\n{name} Metrics:\n")
    results_file.write(f"Mean Squared Error: {mse}\n")
    results_file.write(f"Mean Absolute Error: {mae}\n")
    results_file.write(f"Median Absolute Error: {medae}\n")
    results_file.write(f"Max Error: {maxe}\n")

    # Save scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, predictions)
    plt.title(f'{name} - Predicted vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.xlim([0, max(y_test)+1])
    plt.ylim([0, max(y_test)+1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(f'resultsbrcatcga/{name}_scatter_plot.png')
    plt.close()

results_file.close()

# Feature Importance
feature_importances_rf = best_rf_model.feature_importances_
feature_importances_gb = best_gb_model.feature_importances_

feature_importance_rf_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances_rf}).sort_values(by='Importance', ascending=False)
feature_importance_gb_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances_gb}).sort_values(by='Importance', ascending=False)

feature_importance_rf_df.to_csv("resultsbrcatcga/feature_importance_rf.csv", index=False)
feature_importance_gb_df.to_csv("resultsbrcatcga/feature_importance_gb.csv", index=False)

# Shapley value explanation for Random Forest
explainer_rf = shap.Explainer(best_rf_model)
shap_values_rf = explainer_rf.shap_values(X_test_scaled)
shap.summary_plot(shap_values_rf, X_test_scaled, feature_names=X.columns, show=False)
plt.savefig("resultsbrcatcga/shap_summary_rf.png")
plt.close()

# Shapley value explanation for Gradient Boosting
explainer_gb = shap.Explainer(best_gb_model)
shap_values_gb = explainer_gb.shap_values(X_test_scaled)
shap.summary_plot(shap_values_gb, X_test_scaled, feature_names=X.columns, show=False)
plt.savefig("resultsbrcatcga/shap_summary_gb.png")
plt.close()


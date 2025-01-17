import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Load CSV
data = pd.read_csv('MLcorrelations5.csv')

# Assuming columns "time" (event/censoring time) and "event" (1 if event occurred, 0 if censored) are present
X = data.drop(columns=["OS_MONTHS", "OS_STATUS"])  # features
y = Surv.from_dataframe("OS_STATUS", "OS_MONTHS", data)  # Surv object

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomForestSurvival model
rf_survival = RandomSurvivalForest(random_state=42)

# Hyperparameter optimization using GridSearchCV
param_grid = {
    "n_estimators": [100, 200, 500, 1000],
    "max_depth": [None, 10, 20, 30, 40],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

grid_search = GridSearchCV(rf_survival, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_
best_model

# Model Evaluation
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

# Concordance Index (C-index)
c_index_train = concordance_index_censored(y_train['OS_STATUS'], y_train['OS_MONTHS'], y_pred_train)
c_index_test = concordance_index_censored(y_test['OS_STATUS'], y_test['OS_MONTHS'], y_pred_test)

print(f"Training C-Index: {c_index_train[0]:.4f}")
print(f"Testing C-Index: {c_index_test[0]:.4f}")

# Combine predictions and ground truth from train and test sets
X_combined = pd.concat([X_train, X_test])
y_combined = np.concatenate([y_train, y_test])
y_pred_combined = np.concatenate([y_pred_train, y_pred_test])

# Calculate the global C-index
c_index_global = concordance_index_censored(y_combined['OS_STATUS'], y_combined['OS_MONTHS'], y_pred_combined)

print(f"Global C-Index: {c_index_global[0]:.4f}")

X_test_sorted = X_test.sort_values(by=["rna_PHF16"])
X_test_sel = pd.concat((X_test_sorted.head(2), X_test_sorted.tail(2)))

X_test_sel

pd.Series(best_model.predict(X_test_sel))

surv = best_model.predict_survival_function(X_test_sel, return_array=True)

for i, s in enumerate(surv):
    plt.step(best_model.unique_times_, s, where="post", label=str(i))
plt.ylabel("Survival probability")
plt.xlabel("Time in months")
plt.legend()
plt.grid(True)

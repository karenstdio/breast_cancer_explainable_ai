# Uncomment the following line if LIME is not installed
# !pip install lime

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import shap
import lime.lime_tabular as lime
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Generate synthetic age data between 30 and 80
np.random.seed(42)
age = np.random.randint(30, 81, size=X.shape[0])
X['age'] = age

# Train-test split (including age column separately)
X_train, X_test, y_train, y_test, age_train, age_test = train_test_split(
    X, y, age, test_size=0.2, random_state=42, stratify=y
)

# Standardize numeric features excluding 'age'
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.drop(columns=['age']))
X_test_scaled = scaler.transform(X_test.drop(columns=['age']))

# XGBoost classifier
model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# Performance
y_pred = model.predict(X_test_scaled)
print("Test Accuracy:", round(accuracy_score(y_test, y_pred), 3))

# SHAP explanations
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_scaled)

# SHAP dependence plot for 'mean radius'
shap.dependence_plot(
    ind='mean radius',
    shap_values=shap_values,
    features=X_test.drop(columns=['age']),
    feature_names=data.feature_names
)

# SHAP summary plot
shap.summary_plot(
    shap_values,
    features=X_test.drop(columns=['age']),
    feature_names=X_test.drop(columns=['age']).columns
)

# Group test data by age bins
X_test_with_age = X_test.reset_index(drop=True)
age_bins = pd.cut(age_test, bins=[29, 45, 60, 80], labels=['30-45', '46-60', '61-80'])
X_test_with_age['age_group'] = age_bins

# SHAP value analysis by age group for 'mean radius'
feature_idx = list(X_test.drop(columns=['age']).columns).index('mean radius')
shap_mean_radius = shap_values[:, feature_idx]

df_shap_age = pd.DataFrame({
    'shap_value': shap_mean_radius,
    'age_group': age_bins
})

plt.figure(figsize=(8, 5))
sns.boxplot(data=df_shap_age, x='age_group', y='shap_value')
plt.title("SHAP Value Distribution of 'mean radius' by Age Group")
plt.xlabel("Age Group")
plt.ylabel("SHAP Value (mean radius)")
plt.grid(True)
plt.tight_layout()
plt.show()

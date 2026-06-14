import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, log_loss
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# data loading
df = pd.read_csv('diabetes.csv')
dfcopy = df.copy(deep=True).astype(float)
cols_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
dfcopy[cols_to_impute] = dfcopy[cols_to_impute].replace(0, np.nan)

# slicing
X = dfcopy.drop('Outcome', axis=1)
y = dfcopy['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# imputing the data (using median)
imputer = SimpleImputer(strategy='median')
X_train[cols_to_impute] = imputer.fit_transform(X_train[cols_to_impute])
X_test[cols_to_impute] = imputer.transform(X_test[cols_to_impute])

'''
cols_to_impute_mean = ['Glucose', 'BloodPressure']
cols_to_impute_median = ['SkinThickness', 'Insulin', 'BMI']
X_train[cols_to_impute_mean] = X_train[cols_to_impute_mean].replace(0, np.nan)
X_train[cols_to_impute_median] = X_train[cols_to_impute_median].replace(0, np.nan)
X_test[cols_to_impute_mean] = X_test[cols_to_impute_mean].replace(0, np.nan)
X_test[cols_to_impute_median] = X_test[cols_to_impute_median].replace(0, np.nan)
imputer_med = SimpleImputer(strategy='median')
imputer_mean = SimpleImputer(strategy='mean')
X_train[cols_to_impute_mean] = imputer_mean.fit_transform(X_train[cols_to_impute_mean])
X_train[cols_to_impute_median] = imputer_med.fit_transform(X_train[cols_to_impute_median])
X_test[cols_to_impute_mean] = imputer_mean.transform(X_test[cols_to_impute_mean])
X_test[cols_to_impute_median] = imputer_med.transform(X_test[cols_to_impute_median])
'''

# additional feature creation on train and test separately
X_train['BMI_Cat'] = pd.cut(X_train['BMI'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]).astype(float)
X_train['Glucose_Cat'] = pd.cut(X_train['Glucose'], bins=[0, 90, 140, 200], labels=[0, 1, 2]).astype(float)
X_train['Age_Cat'] = pd.cut(X_train['Age'], bins=[0, 30, 45, 60, 120], labels=[0, 1, 2, 3]).astype(float)
X_train['Glucose_BMI_Ratio'] = X_train['Glucose'] / X_train['BMI']

X_test['BMI_Cat'] = pd.cut(X_test['BMI'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]).astype(float)
X_test['Glucose_Cat'] = pd.cut(X_test['Glucose'], bins=[0, 90, 140, 200], labels=[0, 1, 2]).astype(float)
X_test['Age_Cat'] = pd.cut(X_test['Age'], bins=[0, 30, 45, 60, 120], labels=[0, 1, 2, 3]).astype(float)
X_test['Glucose_BMI_Ratio'] = X_test['Glucose'] / X_test['BMI']


# scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# hyperparameter optimalization using GridSearchCV (using recall metric)
print("Hyperparameter optimalization...\n")

# KNN
params_knn = {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance'], 'p': [1, 2]}
grid_knn = GridSearchCV(KNeighborsClassifier(), params_knn, cv=5, scoring='recall', n_jobs=-1)
grid_knn.fit(X_train_scaled, y_train)
best_knn = grid_knn.best_estimator_

# Decision Tree
params_dt = {'max_depth': [3, 5, 7, None], 'min_samples_split': [2, 5, 10]}
grid_dt = GridSearchCV(DecisionTreeClassifier(class_weight='balanced', random_state=42), params_dt, cv=5, scoring='recall', n_jobs=-1)
grid_dt.fit(X_train_scaled, y_train)
best_dt = grid_dt.best_estimator_

# Boosted Decision Tree (Gradient Boosting)
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

params_bdt = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5]}
grid_bdt = GridSearchCV(GradientBoostingClassifier(random_state=42), params_bdt, cv=5, scoring='recall', n_jobs=-1)
grid_bdt.fit(X_train_scaled, y_train, sample_weight=sample_weights)
best_bdt = grid_bdt.best_estimator_

# LightGBM
params_lgbm = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.05, 0.1], 'num_leaves': [15, 31, 50]}
grid_lgbm = GridSearchCV(lgb.LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1), params_lgbm, cv=5, scoring='recall', n_jobs=-1)
grid_lgbm.fit(X_train_scaled, y_train)
best_lgbm = grid_lgbm.best_estimator_

print("Optimalized parameters:")
print(f"KNN: {grid_knn.best_params_}")
print(f"DT: {grid_dt.best_params_}")
print(f"BDT: {grid_bdt.best_params_}")
print(f"LGBM: {grid_lgbm.best_params_}\n")

# Voting classifier to achieve the best accuracy and recall metrics (different weights because we know some classifiers perform worse on the dataset)
ensemble = VotingClassifier(
    estimators=[('knn', best_knn), ('dt', best_dt), ('bdt', best_bdt), ('lgbm', best_lgbm)],
    voting='soft',
    weights=[1, 1, 2, 2]
)
ensemble.fit(X_train_scaled, y_train)

# Lowered threshold predictions
y_pred_prob = ensemble.predict_proba(X_test_scaled)[:, 1]
custom_threshold = 0.35
y_pred_custom = (y_pred_prob >= custom_threshold).astype(int)

print("Voting Classifier (with a lowered threshold equal to 0.35)")
print(classification_report(y_test, y_pred_custom))

# Accuracy and recall graphs
model_names = ['KNN', 'Decision Tree', 'Boosted DT', 'LightGBM', 'Ensemble (0.5)', 'Ensemble (0.35)']
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860']

acc_scores = [
    accuracy_score(y_test, best_knn.predict(X_test_scaled)),
    accuracy_score(y_test, best_dt.predict(X_test_scaled)),
    accuracy_score(y_test, best_bdt.predict(X_test_scaled)),
    accuracy_score(y_test, best_lgbm.predict(X_test_scaled)),
    accuracy_score(y_test, ensemble.predict(X_test_scaled)),
    accuracy_score(y_test, y_pred_custom)
]

rec_scores = [
    recall_score(y_test, best_knn.predict(X_test_scaled)),
    recall_score(y_test, best_dt.predict(X_test_scaled)),
    recall_score(y_test, best_bdt.predict(X_test_scaled)),
    recall_score(y_test, best_lgbm.predict(X_test_scaled)),
    recall_score(y_test, ensemble.predict(X_test_scaled)),
    recall_score(y_test, y_pred_custom)
]



fig, axes = plt.subplots(1, 2, figsize=(16, 6))

bars_acc = axes[0].bar(model_names, acc_scores, color=colors)
axes[0].set_title('Accuracy comparison')
axes[0].set_ylim(0, 1)
axes[0].tick_params(axis='x', rotation=45)
for bar in bars_acc:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{bar.get_height():.2f}', ha='center', fontweight='bold')

bars_rec = axes[1].bar(model_names, rec_scores, color=colors)
axes[1].set_title('Class 1 recall comparison')
axes[1].set_ylim(0, 1)
axes[1].tick_params(axis='x', rotation=45)
for bar in bars_rec:
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{bar.get_height():.2f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# Accuracy Loss and feature importance
result = permutation_importance(ensemble, X_test_scaled, y_test, scoring='accuracy', n_repeats=10, random_state=42)
plt.figure(figsize=(10, 6))
sns.barplot(x=result.importances_mean, y=X_train.columns, palette='magma', hue=X_train.columns, legend=False)
plt.title('Accuracy Loss - Ensemble')
plt.xlabel('Mean accuracy loss')
plt.tight_layout()
plt.show()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_custom)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Confusion matrix- Ensemble')
plt.show()
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

# Load Dataset


df = pd.read_csv("Churn_Modelling.csv")

# Data Preprocessing

# Remove unnecessary columns
df.drop(['RowNumber', 'CustomerId', 'Surname'],
        axis=1,
        inplace=True)

# Convert categorical data into numerical
df = pd.get_dummies(df, drop_first=True)

# Features and Target

X = df.drop('Exited', axis=1)
y = df['Exited']

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression

lr_model = LogisticRegression()

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("\nLogistic Regression Accuracy")
print(accuracy_score(y_test, lr_pred))

# Random Forest

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Accuracy")
print(accuracy_score(y_test, rf_pred))

# XGBoost

xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

print("\nXGBoost Accuracy")
print(accuracy_score(y_test, xgb_pred))

# Confusion Matrix

print("\nConfusion Matrix")
print(confusion_matrix(y_test, xgb_pred))

# Classification Report

print("\nClassification Report")
print(classification_report(y_test, xgb_pred))

# Save Model

pickle.dump(xgb_model, open("model.pkl", "wb"))

# Save scaler
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("\nModel Saved Successfully")
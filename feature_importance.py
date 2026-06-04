import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

# Load Dataset
df = pd.read_csv("Churn_Modelling.csv")

# Remove unnecessary columns
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# Convert categorical data into numerical
df = pd.get_dummies(df, drop_first=True)

# Features and Target
X = df.drop('Exited', axis=1)
y = df['Exited']

# Train XGBoost Model
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X, y)

# Get Feature Importance
importance = model.feature_importances_

# Feature Names
features = X.columns

# Create Graph
plt.figure(figsize=(10,6))

plt.barh(features, importance)

plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.title("Feature Importance in Customer Churn Prediction")

plt.tight_layout()

# Save graph as image
plt.savefig("feature_importance.png")

# Show graph
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("Churn_Modelling.csv")

# Remove unnecessary columns
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# Convert categorical columns
df = pd.get_dummies(df, drop_first=True)

# Create Figure
plt.figure(figsize=(12,8))

# Heatmap
sns.heatmap(
    df.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.tight_layout()

# Save Image
plt.savefig("heatmap.png")

# Show Graph
plt.show()
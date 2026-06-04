import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("Churn_Modelling.csv")

# Create Figure
plt.figure(figsize=(6,5))

# Churn Distribution
sns.countplot(
    x='Exited',
    data=df
)

plt.title("Customer Churn Distribution")

plt.xlabel("Exited")

plt.ylabel("Count")

plt.tight_layout()

# Save Image
plt.savefig("churn_distribution.png")

# Show Graph
plt.show()
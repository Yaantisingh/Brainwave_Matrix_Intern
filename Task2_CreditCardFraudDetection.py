import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv(r'D:\datasets\creditcard.csv')

# Display the first few rows of the dataset
print(df.head())

# Check for null values
print("Null values in the dataset:\n", df.isnull().sum())

# Handle missing values if present
df = df.dropna()

# Separate the features (X) and the target (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use Isolation Forest for anomaly detection
model = IsolationForest(n_estimators=100, contamination=0.001, random_state=42)
model.fit(X_scaled)

# Predict the anomalies
y_pred = model.predict(X_scaled)

# Map the predictions to the original labels (1 for fraud, -1 for real)
y_pred = np.where(y_pred == -1, 1, 0)

# Evaluate the model
accuracy = accuracy_score(y, y_pred)
conf_mat = confusion_matrix(y, y_pred)
class_report = classification_report(y, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_mat)
print("Classification Report:\n", class_report)

#plotting heatmap for visualization of output
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
fake_news = pd.read_csv(r'D:\datasets\fake.csv')
real_news = pd.read_csv(r'D:\datasets\true.csv')

# Add a label column to both datasets
fake_news['label'] = 1
real_news['label'] = 0

# Combine the datasets
df = pd.concat([fake_news, real_news])

# Shuffle the combined dataset
df = df.sample(frac=1).reset_index(drop=True)

# Display first few rows of the combined dataset
print(df.head())

# Separate the data and labels
X = df['text']
Y = df['label']

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform the training data, transform the testing data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)

# Fit the model on the training data
model.fit(X_train_tfidf, Y_train)

# Predict on the test data
Y_pred = model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print("\n\nAccuracy:", accuracy)

# Print confusion matrix
conf_mat = confusion_matrix(Y_test, Y_pred)
print("\nConfusion Matrix:\n", conf_mat)

# Print classification report
class_report = classification_report(Y_test, Y_pred)
print("\nClassification Report:\n", class_report)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

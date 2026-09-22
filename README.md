import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# Load the Iris dataset
df = load_iris()

# Display raw dataset structure
df

# Convert dataset into a Pandas DataFrame
dfi = pd.DataFrame(df.data, columns=df.feature_names)

# Initialize KNN Classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Extract target labels
y_label = df.target
y_label

# Extract features
features = dfi.iloc[:, 0:4]

# Split dataset into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(features, y_label, test_size=0.2)

# Train the KNN model
knn.fit(x_train, y_train)

# Predict class labels for test data
pv = knn.predict(x_test)

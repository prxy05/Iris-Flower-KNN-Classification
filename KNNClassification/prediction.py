# prediction.py
# Task 6 - KNN Classification using Iris dataset from CSV
# Humanized, clear, and final version with plotting fix

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from matplotlib.colors import ListedColormap

# =========================
# SETTINGS
# =========================
DATA_FILE = "data/Iris.csv"   # path to your dataset
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# STEP 1: Load Dataset
# =========================
df = pd.read_csv(DATA_FILE)

# Drop Id column if it exists
if "Id" in df.columns:
    df = df.drop(columns=["Id"])

# Features and target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Unique class names
class_names = np.unique(y)

print("✅ Dataset loaded successfully.")
print("Shape:", X.shape)
print("Classes:", class_names)

# =========================
# STEP 2: Normalize Features
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\n🔄 Features normalized.")

# =========================
# STEP 3: Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print("📊 Train size:", X_train.shape, "Test size:", X_test.shape)

# =========================
# STEP 4: Experiment with different K values
# =========================
k_values = range(1, 21)
train_accuracies = []
test_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_accuracies.append(knn.score(X_train, y_train))
    test_accuracies.append(knn.score(X_test, y_test))

# Save accuracy plot
plt.figure(figsize=(8, 5))
plt.plot(k_values, train_accuracies, marker='o', label='Train Accuracy')
plt.plot(k_values, test_accuracies, marker='o', label='Test Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy for different K values')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "knn_k_vs_accuracy.png"))
plt.close()

best_k = k_values[np.argmax(test_accuracies)]
print("\n✅ Best K value:", best_k, "with Test Accuracy:", max(test_accuracies))

# =========================
# STEP 5: Train final KNN model with best K
# =========================
knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train, y_train)
y_pred = knn_final.predict(X_test)

print("\n📈 Final KNN Model Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=class_names))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# =========================
# STEP 6: Visualize Decision Boundaries (first 2 features only)
# =========================
X_vis = X_scaled[:, :2]  # only first 2 features for 2D plot
X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(
    X_vis, y, test_size=0.2, random_state=42, stratify=y
)

knn_vis = KNeighborsClassifier(n_neighbors=best_k)
knn_vis.fit(X_train_vis, y_train_vis)

h = .02  # mesh step size
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict and convert class names to numeric codes for plotting
Z = knn_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = pd.factorize(Z)[0]  # convert to integer labels
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

plt.contourf(xx, yy, Z, cmap=cmap_light)
plt.scatter(X_vis[:, 0], X_vis[:, 1],
            c=pd.factorize(y)[0], cmap=cmap_bold, edgecolor='k', s=40)
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.title(f"KNN Decision Boundary (K={best_k}) - First 2 Features")
plt.savefig(os.path.join(OUTPUT_DIR, "knn_decision_boundary.png"))
plt.close()

print("\n🏁 All done! Plots saved in:", OUTPUT_DIR)

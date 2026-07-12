import os
import cv2
import numpy as np

# Load images
data = []
labels = []

# Free = 0
for file in os.listdir("dataset/free"):
    img = cv2.imread(os.path.join("dataset/free", file))
    img = cv2.resize(img, (32, 32))
    data.append(img.flatten())
    labels.append(0)

# Occupied = 1
for file in os.listdir("dataset/occupied"):
    img = cv2.imread(os.path.join("dataset/occupied", file))
    img = cv2.resize(img, (32, 32))
    data.append(img.flatten())
    labels.append(1)

X = np.array(data)
y = np.array(labels)

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create model
lr_model = LogisticRegression(max_iter=1000)

# Train model
lr_model.fit(X_train, y_train)

# Predictions
y_pred = lr_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Logistic Regression Accuracy:", accuracy)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Predictions
y_pred = lr_model.predict(X_test)
y_prob = lr_model.predict_proba(X_test)[:, 1]

print("\nLogistic Regression Results")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob))

from sklearn.ensemble import RandomForestClassifier

# Random Forest Model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

# Predictions
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

print("\nRandom Forest Results")
print("Accuracy :", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall   :", recall_score(y_test, rf_pred))
print("F1 Score :", f1_score(y_test, rf_pred))
print("ROC-AUC  :", roc_auc_score(y_test, rf_prob))

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    rf_model,
    X,
    y,
    cv=5,
    scoring='accuracy'
)

print("\nCross Validation Scores:")
print(cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, rf_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label="Random Forest (AUC = 0.987)")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)

plt.show()

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6,5))
plt.imshow(cm, cmap='Blues')

plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks([0,1], ["Free", "Occupied"])
plt.yticks([0,1], ["Free", "Occupied"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j],
                 ha="center",
                 va="center",
                 color="black",
                 fontsize=14)

plt.colorbar()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Metrics
metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]

# Scores
logistic = [0.7722, 0.8621, 0.8000, 0.8299, 0.7831]
random_forest = [0.9444, 0.9323, 0.9920, 0.9612, 0.9868]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10,6))

plt.bar(x - width/2, logistic, width, label="Logistic Regression")
plt.bar(x + width/2, random_forest, width, label="Random Forest")

plt.xticks(x, metrics)
plt.ylabel("Score")
plt.xlabel("Evaluation Metrics")
plt.title("Performance Comparison of Logistic Regression and Random Forest")
plt.ylim(0, 1.05)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Display values on top of bars
for i, value in enumerate(logistic):
    plt.text(i - width/2, value + 0.01, f"{value:.2f}", ha='center', fontsize=9)

for i, value in enumerate(random_forest):
    plt.text(i + width/2, value + 0.01, f"{value:.2f}", ha='center', fontsize=9)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

labels = ["Free", "Occupied"]
sizes = [273, 624]

plt.figure(figsize=(6,6))
plt.pie(
    sizes,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Dataset Distribution")
plt.show()
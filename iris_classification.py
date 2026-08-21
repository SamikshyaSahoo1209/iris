"""
Task 3: Iris Flower Classification
------------------------------------
Trains a machine learning model to classify Iris flowers into
setosa, versicolor, or virginica based on sepal and petal measurements.

Dataset: The classic Iris dataset (available built-in via scikit-learn,
or downloadable from the UCI Machine Learning Repository).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ---------------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------------
iris = load_iris(as_frame=True)
df = iris.frame  # includes feature columns + numeric 'target'
df["species"] = df["target"].apply(lambda i: iris.target_names[i])

print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nClass distribution:\n", df["species"].value_counts())

# If you'd rather load from a CSV downloaded from UCI/Kaggle instead of
# sklearn's built-in copy, comment the block above and use:
# df = pd.read_csv("Iris.csv")
# df.rename(columns={"Species": "species"}, inplace=True)


# ---------------------------------------------------------
# 2. Quick exploratory data analysis (optional but useful)
# ---------------------------------------------------------
sns.pairplot(df, hue="species", vars=iris.feature_names)
plt.suptitle("Iris Feature Relationships by Species", y=1.02)
plt.savefig("iris_pairplot.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(df[iris.feature_names].corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig("iris_correlation.png", dpi=150, bbox_inches="tight")
plt.close()


# ---------------------------------------------------------
# 3. Prepare features and target
# ---------------------------------------------------------
X = df[iris.feature_names]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (helps distance-based / gradient-based models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------------
# 4. Train and compare several models
# ---------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine": SVC(kernel="rbf", probability=True),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    # 5-fold cross-validation on the training set for a more robust estimate
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

    results[name] = {
        "test_accuracy": acc,
        "cv_mean_accuracy": cv_scores.mean(),
        "cv_std": cv_scores.std(),
    }

    print(f"\n=== {name} ===")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Cross-Val Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))


# ---------------------------------------------------------
# 5. Pick the best model and show its confusion matrix
# ---------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["test_accuracy"])
best_model = models[best_name]
print(f"\nBest performing model: {best_name} "
      f"(Test Accuracy: {results[best_name]['test_accuracy']:.4f})")

y_pred_best = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix - {best_name}")
plt.savefig("iris_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()


# ---------------------------------------------------------
# 6. Summary table of all models
# ---------------------------------------------------------
summary_df = pd.DataFrame(results).T.sort_values("test_accuracy", ascending=False)
print("\nModel Comparison Summary:")
print(summary_df)


# ---------------------------------------------------------
# 7. Example: predict species for a new, unseen flower measurement
# ---------------------------------------------------------
sample = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],  # sepal_length, sepal_width, petal_length, petal_width
    columns=iris.feature_names,
)
sample_scaled = scaler.transform(sample)
prediction = best_model.predict(sample_scaled)[0]
print(f"\nSample measurement {sample.values.tolist()[0]} "
      f"predicted as: {iris.target_names[prediction]}")

import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

# --------------------------------
# Load Test Data
# --------------------------------
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()

# --------------------------------
# Load Saved Model
# --------------------------------
with open("models/attrition_model.pkl", "rb") as file:
    model = pickle.load(file)

print("Model Loaded Successfully!")

# --------------------------------
# Predictions
# --------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# --------------------------------
# Evaluation Metrics
# --------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# --------------------------------
# Classification Report
# --------------------------------
report = classification_report(
    y_test,
    y_pred,
    target_names=["Stay", "Leave"]
)

print("\nClassification Report:")
print(report)

# SAVE CLASSIFICATION REPORT
with open("reports/classification_report.txt", "w") as file:
    file.write(report)

print("Classification report saved successfully!")

# --------------------------------
# Confusion Matrix
# --------------------------------
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Stay", "Leave"]
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.savefig(
    "images/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# --------------------------------
# ROC Curve
# --------------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)

auc_score = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6, 5))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc_score:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()

plt.savefig(
    "images/roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# --------------------------------
# Feature Importance
# --------------------------------
# This section is for Logistic Regression
importance = model.coef_[0]

feature_importance = pd.DataFrame({
    "Feature": X_test.columns,
    "Coefficient": importance,
    "Importance": abs(importance)
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(
    feature_importance[
        ["Feature", "Coefficient"]
    ].head(10)
)

top_features = feature_importance.head(10)

plt.figure(figsize=(8, 5))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.gca().invert_yaxis()

plt.title("Top 10 Important Features")

plt.savefig(
    "images/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# --------------------------------
# Save Metrics
# --------------------------------
metrics_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        auc_score
    ]
})

metrics_df.to_csv(
    "reports/model_metrics.csv",
    index=False
)

print("\nMetrics:")
print(metrics_df)

print("\nEvaluation completed successfully!")


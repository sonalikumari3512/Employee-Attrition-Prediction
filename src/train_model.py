import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score


X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")

Y_train = pd.read_csv("data/Y_train.csv").squeeze()
Y_test = pd.read_csv("data/Y_test.csv").squeeze()

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)


#train logistic Regression
log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train,Y_train)

log_pred = log_model.predict(X_test)

log_accuracy = accuracy_score(Y_test,log_pred)
print("Logistic Regression Accuracy:", log_accuracy)


#train Decision tree
tree_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)

tree_model.fit(X_train, Y_train)

tree_pred = tree_model.predict(X_test)

tree_accuracy = accuracy_score(Y_test, tree_pred)

print("Decision Tree Accuracy:", tree_accuracy)

#train random forest

forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

forest_model.fit(X_train, Y_train)

forest_pred = forest_model.predict(X_test)

forest_accuracy = accuracy_score(Y_test, forest_pred)

print("Random Forest Accuracy:", forest_accuracy)


results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "Accuracy": [
        log_accuracy,
        tree_accuracy,
        forest_accuracy
    ]
})

results = results.sort_values(
    by="Accuracy",
    ascending=False
)

print(results)


models = {
    "Logistic Regression": log_model,
    "Decision Tree": tree_model,
    "Random Forest": forest_model
}

accuracies = {
    "Logistic Regression": log_accuracy,
    "Decision Tree": tree_accuracy,
    "Random Forest": forest_accuracy
}

best_model_name = max(
    accuracies,
    key=accuracies.get
)

best_model = models[best_model_name]

print("Best Model:", best_model_name)
print("Accuracy:", accuracies[best_model_name])

with open("models/attrition_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

print("Best model saved successfully!")

#test best model

with open("models/attrition_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

prediction = loaded_model.predict(X_test.iloc[[0]])

print("\nSample Prediction:", prediction[0])
print("Actual Label:", Y_test.iloc[0])
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler


df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

print(df.shape)
print(df.head())

print(df["Attrition"].value_counts())

label_encoder =LabelEncoder()

df["Attrition"] = label_encoder.fit_transform(df["Attrition"])
print(df["Attrition"].value_counts())

constant_columns = []

for column in df.columns:
    if df[column].nunique() == 1:
        constant_columns.append(column)

print(constant_columns)

df.drop(columns=constant_columns,inplace=True)
print(df.shape)

X = df.drop("Attrition",axis=1)

Y = df["Attrition"]

print(X.shape)
print(Y.shape)

categorical_columns = X.select_dtypes(
    include=["object", "string"]
).columns

print(categorical_columns)

#one hot encoding

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True
)

print(X.shape)

X_train,X_test,Y_train,Y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42,
    stratify=Y
)

print("Training:", X_train.shape)
print("Testing :", X_test.shape)

#scale numerical features
numeric_columns = X_train.select_dtypes(include=["int64", "float64"]).columns

scaler = StandardScaler()

X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

print(X_train[numeric_columns].head())

##save processed data
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)

Y_train.to_csv("data/Y_train.csv", index=False)
Y_test.to_csv("data/Y_test.csv", index=False)

print("Processed datasets saved successfully!")

import pickle

with open("models/scaler.pkl","wb") as file:
    pickle.dump(scaler,file)


print("Scaler saved successfully!")
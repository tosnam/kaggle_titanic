# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # How to do the Titanic Kaggle Competition
# - https://www.youtube.com/watch?v=pUSi5xexT4Q&ab_channel=AladdinPersson

import pandas as pd

data = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
test_ids = test["PassengerId"]

data.head(5)

data.Embarked.value_counts()


def clean(data):
    data = data.drop(["Ticket", "Cabin", "Name", "PassengerId"], axis=1)

    cols = ["SibSp", "Parch", "Fare", "Age"]
    for col in cols:
        data[col].fillna(data[col].median(), inplace=True)

    data.Embarked.fillna("U", inplace=True)
    return data


data = clean(data)
test = clean(test)

data.head(5)

# +
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

cols = ["Sex", "Embarked"]

for col in cols:
    data[col] = le.fit_transform(data[col])
    test[col] = le.transform(test[col])
    print(le.classes_)
# -

data.head(5)

# +
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

y = data["Survived"]
X = data.drop("Survived", axis=1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# -

y_train

clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)

predictions = clf.predict(X_val)
from sklearn.metrics import accuracy_score
accuracy_score(y_val, predictions)

submission_preds = clf.predict(test)

df = pd.DataFrame({"PassengerId":test_ids.values,
                   "Survived":submission_preds,
                  })

df.to_csv("submission.csv", index=False)

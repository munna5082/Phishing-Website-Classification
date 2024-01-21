import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import pickle

df = pd.read_csv("Phishing.csv")
print(df.head(5))

df.drop(columns=['id'], inplace=True)
print(df.head(5))
print(df.shape)
print(df.isna().sum())
print(df.describe())

print(df["CLASS_LABEL"].value_counts())

cor = df.corr()
plt.figure(figsize=(100, 100))
sns.heatmap(cor, annot=True, cmap="bone", linewidths=.5)
plt.show()

print(cor["CLASS_LABEL"].sort_values())


features = df.drop(columns="CLASS_LABEL")
target = df["CLASS_LABEL"]

feature_selector = ExtraTreesClassifier(n_estimators=5)
feature_selector.fit(features, target)
feature_importances = feature_selector.feature_importances_
s1 = pd.Series(feature_importances)
s1.index = features.columns
print(s1.sort_values(ascending=False))

feature_importances_normalized = np.std([tree.feature_importances_ for tree in feature_selector.estimators_], axis = 0)
plt.figure(figsize=(20, 20))
plt.bar(features.columns, feature_importances_normalized)
plt.xlabel('Features')
plt.ylabel('Feature Impotances')
plt.xticks(rotation=90)
plt.show()

obj = mutual_info_classif(features, target)
mutual_info = pd.Series(obj)
mutual_info.index = features.columns
print(mutual_info.sort_values(ascending=False))
mutual_info.sort_values(ascending=False).plot.bar(figsize=(20, 10))
plt.show()

cols_to_use = ['NumDots', 'PathLevel', 'NumDash', 'NumSensitiveWords', 'PctExtHyperlinks', 'PctExtResourceUrls', 'InsecureForms', 'PctNullSelfRedirectHyperlinks',
               'FrequentDomainNameMismatch', 'SubmitInfoToEmail', 'IframeOrFrame']

X = df[cols_to_use]
y = df['CLASS_LABEL']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


rfc_model = RandomForestClassifier(n_estimators=40)
svc_model = SVC()
kneighbors_model = KNeighborsClassifier(n_neighbors=5)
naivebayes_model = GaussianNB()
xgboost_model = XGBClassifier()

models = [rfc_model, svc_model, kneighbors_model, naivebayes_model, xgboost_model]
for model in models:
    scores = cross_val_score(model, X_train, y_train, cv=5)
    avg = scores.mean()
    print(model, ':', avg)

rfc_model.fit(X_train, y_train)
y_pred = rfc_model.predict(X_test)

print(rfc_model.score(X_test, y_test))
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(f1_score(y_test, y_pred))

with open('websiteclassification.pkl', 'wb')as file:
    pickle.dump(rfc_model, file)
    file.close()

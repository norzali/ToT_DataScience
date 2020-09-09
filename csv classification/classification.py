#Importing Libraries
import itertools
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
#%matplotlib inline

#Importing data
#!wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv
df = pd.read_csv('C:/Users/Lenovo/Documents/ToT_DataScience/csv classification/loan_train.csv')
df.head()
df.shape

#Convert to date time object#

df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()
df['loan_status'].value_counts()

#!conda install -c anaconda seaborn -y

import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()

bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()

#Pre-processing: Feature selection/extraction#

df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()

#Convert Categorical features to numerical values#
df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)

df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


df.groupby(['education'])['loan_status'].value_counts(normalize=True)

df[['Principal','terms','age','Gender','education']].head()

Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()

X = Feature
X[0:5]



y = df['loan_status'].values
y[0:5]

#Normalize Data#

X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.2, random_state=4)

#K Nearest Neighbor(KNN)#

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_pred_proba = classifier.predict_proba(X_test)


from sklearn import metrics
print('Train set accuracy:', metrics.accuracy_score(y_train, classifier.predict(X_train)))
print('test set accuracy:', metrics.accuracy_score(y_test, y_pred))

#Decision Tree#

from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4)
clf.fit(X_train, y_train)

Predtree = clf.predict(X_test)
Predtree_proba = clf.predict_proba(X_test)


#Support Vector Machine#
from sklearn import svm
clf_SVM = svm.SVC(kernel='rbf')
clf_SVM.fit(X_train, y_train)
clf_SVM

from sklearn import svm
clf_SVM = svm.SVC(kernel='rbf')
clf_SVM.fit(X_train, y_train)
clf_SVM

yhat = clf_SVM.predict(X_test)
yhat

#Logistic Regression#
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression().fit(X_train, y_train)
LR

y_hat = LR.predict(X_test)
y_hat[0:5]
y_hat_proba = LR.predict_proba(X_test)

#Model Evaluation using Test set#

from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

test_df = pd.read_csv('C:/Users/Lenovo/Documents/ToT_DataScience/csv classification/loan_test.csv')
test_df.head()

#Jaccard Similarity Score#

KNN_J_Accuracy = jaccard_score(y_test, y_pred)
DT_J_Accuracy = jaccard_score(y_test, Predtree)
SVM_J_Accuracy = jaccard_score(y_test, yhat)
LR_J_Accuracy = jaccard_score(y_test, y_hat)

J_Score = np.array([KNN_J_Accuracy, DT_J_Accuracy, SVM_J_Accuracy, LR_J_Accuracy])
J_Score

#F1 Score#

KNN_F1 = f1_score(y_test, y_pred, average= 'weighted')
DT_F1 = f1_score(y_test, Predtree, average= 'weighted')
SVM_F1 = f1_score(y_test, yhat, average= 'weighted')
LR_F1 = f1_score(y_test, y_hat, average= 'weighted')

F1_scores = np.array([KNN_F1, DT_F1, SVM_F1, LR_F1])
F1_scores

#Log_loss#
LR_Log_loss = log_loss(y_test, y_hat_proba)
LR_Log_loss

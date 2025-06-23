import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

df = pd.read_csv(r"C:\Users\LENOVO\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv")
# print(df.head(5))
# print(df.info())

#Data Preprocessing

#Data type error in total charger we need to convert to numerical values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# print(df.info())

#Handling missing values
#Total Charges have 11 missing values and its missing values is 0.15% which is less than 1% so , we get rid of them

df.dropna(how='any', inplace=True)

#Divide the data into Features and Target
X= df.drop(['customerID', 'Churn'], axis=1)
y = df.Churn.values

print(df.Churn.value_counts()/len(df)*100)
# print(X.columns)

#Feature Encoding--> Converting the categorical featuures to numerical
X = pd.get_dummies(X, columns=['gender', 'Partner', 'Dependents',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'],drop_first=True, dtype=int)

#Spliting the data into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

#Feature Scaling
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

#KNN Classifier
model_knn = KNeighborsClassifier()
model_knn.fit(X_train_sc,y_train)
y_pred_knn = model_knn.predict(X_test_sc)

# Model Evalution
accuracy_knn = round(accuracy_score(y_test,y_pred_knn)*100,2)
print("KNN accuracy",accuracy_knn) #76.05

#Decision Tree
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train_sc,y_train)
y_pred_dt = model_dt.predict(X_test_sc)

# Model Evalution
accuracy_dt = round(accuracy_score(y_test,y_pred_dt)*100,2)
print("Decision Tree accuracy",accuracy_dt) #72.01

#Random Forest 
model_rf = RandomForestClassifier(n_estimators=200)
model_rf.fit(X_train_sc,y_train)
y_pred_rf = model_rf.predict(X_test_sc)

# Model Evalution
accuracy_rf = round(accuracy_score(y_test,y_pred_rf)*100,2)
print("Random Forest",accuracy_rf) # 79.01

#Navie Bayes classifier
model_nb = BernoulliNB()
model_nb.fit(X_train_sc,y_train)
y_pred_nb = model_nb.predict(X_test_sc)

# Model Evalution
accuracy_nb = round(accuracy_score(y_test,y_pred_nb)*100,2)
print("Naive Bayes",accuracy_nb) 

#Fit the model
model_svm = SVC(kernel='linear')
model_svm.fit(X_train_sc,y_train)
y_pred_svm = model_svm.predict(X_test_sc)

# Model Evalution
accuracy_svm = round(accuracy_score(y_test,y_pred_svm)*100,2)
print("SVM ",accuracy_svm) 

#Logistic Regression
model_log = LogisticRegression()
model_log.fit(X_train_sc,y_train)
y_pred_log = model_log.predict(X_test_sc)

# Model Evalution
accuracy_log = round(accuracy_score(y_test,y_pred_log)*100,2)
print("Logistic Regression ",accuracy_log) 

print("KNN",classification_report(y_test,y_pred_knn))
print("decision",classification_report(y_test,y_pred_dt))
print("Random",classification_report(y_test,y_pred_rf))
print("Naive Bayes",classification_report(y_test,y_pred_nb))
print("SVM",classification_report(y_test,y_pred_svm))
print("Logistics",classification_report(y_test,y_pred_log))

#Comparision
'''Classifiers	Accuracy	Recall	Precission	F1_Score
KNN	       75.88	       0.51	0.52	       0.52
Decision Tree	72.92	       0.47	0.46	       0.47
Random_Forest	79.86	       0.49	0.63	       0.55
Naïve Bayes	71.62	       0.83	0.47	       0.60
SVM	       80.49	       0.54	0.63	       0.58
Logistics	80.66	       0.62	0.61	       0.62 '''

print("KNN \n",confusion_matrix(y_test,y_pred_knn))
print("decision \n",confusion_matrix(y_test,y_pred_dt))
print("Random \n",confusion_matrix(y_test,y_pred_rf))
print("Naive Bayes \n",confusion_matrix(y_test,y_pred_nb))
print("SVM \n",confusion_matrix(y_test,y_pred_svm))
print("Logistics \n",confusion_matrix(y_test,y_pred_log))

'''
KNN
[[1094  182]
  [ 239  243]]
decision
 [[1051  236]
 [ 242  229]]
Random
 [[1170  117]
 [ 247  224]]
Naive Bayes
 [[927 360]
 [ 92 379]]
SVM
 [[1157  130]
 [ 225  246]]
Logistics
 [[1156  131]
 [ 223  248]] '''
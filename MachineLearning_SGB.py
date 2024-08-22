import pandas as pd
import re
import numpy as np
import csv
import imblearn



from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics._classification import classification_report
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import statsmodels.api as sm
from imblearn.over_sampling import SMOTE
import math
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier

#Read the CSV file
df=pd.read_csv("Storm_reduceddata_csv.csv")
print(df.columns)
#check the imbalance
print(df['Output'].value_counts())
print(df.Output.value_counts()/len(df))
#Independent Variables
X=df.drop(columns=['Output'])
#Dependent Variable
y=df[['Output']]

#SMOTE sampling
os=SMOTE(random_state=0)
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.25, random_state=100)
columns=X_train.columns

smote_data_X, smote_data_y= os.fit_resample(X_train, y_train)
smote_data_X=pd.DataFrame(data=smote_data_X,columns=columns)
smote_data_y=pd.DataFrame(data=smote_data_y,columns=['Output'])
print(smote_data_X)
print(smote_data_y)

#compare the class counts in the original and SMOTE dataset
print('% of each class in the original dataset-')
print(df.Output.value_counts()/len(df))
print('% of each class in the SMOTE dataset-')
print(smote_data_y.Output.value_counts()/len(df))
#create X and y after SMOTE is applied
X=smote_data_X[columns]
y=smote_data_y['Output']


#Split the dataset to test and train data and applied Naive Bayes
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.25, random_state=0)
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)
print('Scaling Done')
model=GradientBoostingClassifier(n_estimators=100, random_state = 5)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print('Predicted Class:%d' %y_pred[0])

#performance measures
cm=confusion_matrix(y_test, y_pred)
print(cm)
sns.heatmap(cm,annot=True)
plt.savefig('confusion.png')
precision=precision_score(y_test, y_pred)
print('Precision:%.3f' % precision)
recall=recall_score(y_test, y_pred)
print('Recall:%.3f' % recall)
Accuracy=accuracy_score(y_test,y_pred)
print('Accuracy:%.3f' %Accuracy)
score=f1_score(y_test, y_pred)
print('F-measure:%.3f' % score)

#ROC Curve
Bagging_roc_auc = roc_auc_score(y_test,y_pred)
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Bagging with KNN (area = %0.2f)' % Bagging_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Bagging_ROC')
plt.show()
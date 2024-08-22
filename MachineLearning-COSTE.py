import pandas as pd
import re
import numpy as np
import csv
import imblearn



from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics._classification import classification_report
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
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold


#Read the CSV file
datasetTrain = pd.read_csv('hive_Balanced_File.csv')


datasetTest = pd.read_csv('hive_Balanced_File.csv')


#check the imbalance
print(datasetTrain['Output'].value_counts())
print(datasetTrain.Output.value_counts()/len(datasetTrain))
#Independent Variables
X_tr=datasetTrain.drop(columns=['Output'])
#X_tr=datasetTrain.drop(index=datasetTrain.index[0], axis=0, inplace=True)
#Dependent Variable
y_tr=datasetTrain[['Output']]


print('------Splitting------')
#Split the Training dataset
X_train, X_test, y_train, y_test= train_test_split(X_tr,y_tr,test_size=0.3, random_state=0)
print('Training data',X_train.shape)
print('Training y',y_train.shape)
print('Testing data',X_test.shape)
print('Testing y',y_test.shape)



print('-----Testing dataset-----')
#check the imbalance
print(datasetTest['Output'].value_counts())
print(datasetTest.Output.value_counts()/len(datasetTest))
#Independent Variables
X_tst=datasetTest.drop(columns=['Output'])
#X_tst=datasetTest.drop(index=datasetTest.index[0], axis=0, inplace=True)
#Dependent Variable
y_tst=datasetTest[['Output']]



#Split the Testing  dataset 
X_tr_tst, X_test_tst, y_train_tst, y_test_tst= train_test_split(X_tst,y_tst,test_size=0.3, random_state=0)
print('Training data',X_tr_tst.shape)
print('Training y',y_train_tst.shape)
print('Testing data',X_test_tst.shape)
print('Testing y',y_test_tst.shape)


#------------------DECISION TREE CLASSIFIER----------------------------
print('Decision Tree Classifier')
#Split the dataset to test and train data and applied Decision Tree Classifier
#X_train, X_test, y_train, y_test= train_test_split(X_tr,y_tr,test_size=0.3, random_state=0)
model=DecisionTreeClassifier()
#Evaluate Pipeline
cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores=cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
model.fit(X_train,y_train)
pred=model.predict(X_test_tst)
print(accuracy_score(y_test_tst, pred))
# print("Classifier Accuracy {:.2f} %".format(model.score(X_test,y_test)*100))
#performance measures
cm=confusion_matrix(y_test_tst, pred)
print(cm)
sns.heatmap(cm,annot=True)
plt.savefig('confusion.png')
prec=precision_score(y_test_tst, pred)
print('Precision:%.3f' % prec)
recall=recall_score(y_test_tst, pred)
print('Recall:%.3f' % recall)
Accuracy=accuracy_score(y_test_tst,pred)
print('Accuracy:%.3f' %Accuracy)
score=f1_score(y_test_tst, pred)
print('F-measure:%.3f' % score)

# Creating a DataFrame
DataSample= [[prec,recall,Accuracy,score]]
SimpleDataFrame=pd.DataFrame(data=DataSample, columns=['Precision','Recall','Accuracy','F1-Score'])
print(SimpleDataFrame)
SimpleDataFrame.name="Decision Tree Classifier"
 


#-----------------LOGISTIC REGRESSION----------------------------------
print('Logistic Regression')
model=LogisticRegression(solver='liblinear', random_state=0)
#logreg=LogisticRegression()
model.fit(X_train,y_train)
pred=model.predict(X_test_tst)
print(accuracy_score(y_test_tst, pred))
# print("Classifier Accuracy {:.2f} %".format(model.score(X_test,y_test)*100))
#performance measures
cm=confusion_matrix(y_test_tst, pred)
print(cm)
sns.heatmap(cm,annot=True)
plt.savefig('confusion.png')
prec=precision_score(y_test_tst, pred)
print('Precision:%.3f' % prec)
recall=recall_score(y_test_tst, pred)
print('Recall:%.3f' % recall)
Accuracy=accuracy_score(y_test_tst,pred)
print('Accuracy:%.3f' %Accuracy)
score=f1_score(y_test_tst, pred)
print('F-measure:%.3f' % score)
# Creating a DataFrame
DataSample2= [[prec,recall,Accuracy,score]]
SimpleDataFrame2=pd.DataFrame(data=DataSample2, columns=['Precision','Recall','Accuracy','F1-Score'])
SimpleDataFrame2.name="Logistic regression"
print(SimpleDataFrame2)


#------------Naive Bayes--------------------------
print('Naive Bayes')
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
#create a Gaussian Classifier
classifier1=GaussianNB()
classifier1.fit(X_train,y_train)
pred=classifier1.predict(X_test_tst)
print(accuracy_score(y_test_tst, pred))
# print("Classifier Accuracy {:.2f} %".format(model.score(X_test,y_test)*100))
#performance measures
cm=confusion_matrix(y_test_tst, pred)
print(cm)
sns.heatmap(cm,annot=True)
plt.savefig('confusion.png')
prec=precision_score(y_test_tst, pred)
print('Precision:%.3f' % prec)
recall=recall_score(y_test_tst, pred)
print('Recall:%.3f' % recall)
Accuracy=accuracy_score(y_test_tst,pred)
print('Accuracy:%.3f' %Accuracy)
score=f1_score(y_test_tst, pred)
print('F-measure:%.3f' % score)
# Creating a DataFrame
DataSample6= [[prec,recall,Accuracy,score]]
SimpleDataFrame6=pd.DataFrame(data=DataSample6, columns=['Precision','Recall','Accuracy','F1-Score'])
print(SimpleDataFrame6)
SimpleDataFrame6.name="Naive Bayes"

#-------------Bagging------------
print('Bagging')
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
model=BaggingClassifier(base_estimator=KNeighborsClassifier())
model.fit(X_train,y_train)
pred=model.predict(X_test_tst)
print(accuracy_score(y_test_tst, pred))
# print("Classifier Accuracy {:.2f} %".format(model.score(X_test,y_test)*100))
#performance measures
cm=confusion_matrix(y_test_tst, pred)
print(cm)
sns.heatmap(cm,annot=True)
plt.savefig('confusion.png')
prec=precision_score(y_test_tst, pred)
print('Precision:%.3f' % prec)
recall=recall_score(y_test_tst, pred)
print('Recall:%.3f' % recall)
Accuracy=accuracy_score(y_test_tst,pred)
print('Accuracy:%.3f' %Accuracy)
score=f1_score(y_test_tst, pred)
print('F-measure:%.3f' % score)
# Creating a DataFrame
DataSample3= [[prec,recall,Accuracy,score]]
SimpleDataFrame3=pd.DataFrame(data=DataSample3, columns=['Precision','Recall','Accuracy','F1-Score'])
print(SimpleDataFrame3)
SimpleDataFrame3.name="Bagging"


#-------------Adaboost -----------------
print('Adaboost')
from sklearn.ensemble import AdaBoostClassifier
abc=AdaBoostClassifier(n_estimators=50, learning_rate=1)
model=abc.fit(X_train,y_train)
pred=model.predict(X_test_tst)
print(accuracy_score(y_test_tst, pred))
# print("Classifier Accuracy {:.2f} %".format(model.score(X_test,y_test)*100))
#performance measures
cm=confusion_matrix(y_test_tst, pred)
print(cm)
sns.heatmap(cm,annot=True)
plt.savefig('confusion.png')
prec=precision_score(y_test_tst, pred)
print('Precision:%.3f' % prec)
recall=recall_score(y_test_tst, pred)
print('Recall:%.3f' % recall)
Accuracy=accuracy_score(y_test_tst,pred)
print('Accuracy:%.3f' %Accuracy)
score=f1_score(y_test_tst, pred)
print('F-measure:%.3f' % score)
# Creating a DataFrame
DataSample4= [[prec,recall,Accuracy,score]]
SimpleDataFrame4=pd.DataFrame(data=DataSample4, columns=['Precision','Recall','Accuracy','F1-Score'])
print(SimpleDataFrame4)
SimpleDataFrame4.name="Adaboost"

#-------Random Forest--------------
print('Random Forest')
from sklearn.ensemble._forest import RandomForestClassifier
#Instantiate model with 1000 decision trees
rf=RandomForestClassifier()
#Train the model on training data
model= rf.fit(X_train,y_train)
#use the forest predict method on test data
pred=model.predict(X_test_tst)
print(accuracy_score(y_test_tst, pred))
# print("Classifier Accuracy {:.2f} %".format(model.score(X_test,y_test)*100))
#performance measures
cm=confusion_matrix(y_test_tst, pred)
print(cm)
sns.heatmap(cm,annot=True)
plt.savefig('confusion.png')
prec=precision_score(y_test_tst, pred)
print('Precision:%.3f' % prec)
recall=recall_score(y_test_tst, pred)
print('Recall:%.3f' % recall)
Accuracy=accuracy_score(y_test_tst,pred)
print('Accuracy:%.3f' %Accuracy)
score=f1_score(y_test_tst, pred)
print('F-measure:%.3f' % score)
# Creating a DataFrame
DataSample5= [[prec,recall,Accuracy,score]]
SimpleDataFrame5=pd.DataFrame(data=DataSample5, columns=['Precision','Recall','Accuracy','F1-Score'])
print(SimpleDataFrame5)
SimpleDataFrame5.name="Random Forest"

#------Exporting the Data into Excel File------------
writer = pd.ExcelWriter('test.xlsx',engine='xlsxwriter')
workbook=writer.book
worksheet=workbook.add_worksheet('Result')
writer.sheets['Result'] = worksheet

worksheet.write_string(0, 0, SimpleDataFrame.name)
SimpleDataFrame.to_excel(writer,sheet_name='Result',startrow=1 , startcol=0)

worksheet.write_string(SimpleDataFrame.shape[0] + 4, 0, SimpleDataFrame2.name)
SimpleDataFrame2.to_excel(writer,sheet_name='Result',startrow=SimpleDataFrame.shape[0] + 5, startcol=0)

worksheet.write_string(SimpleDataFrame2.shape[0] + 8, 0, SimpleDataFrame3.name)
SimpleDataFrame3.to_excel(writer,sheet_name='Result',startrow=SimpleDataFrame2.shape[0] + 9, startcol=0)

worksheet.write_string(SimpleDataFrame3.shape[0] + 12, 0, SimpleDataFrame4.name)
SimpleDataFrame4.to_excel(writer,sheet_name='Result',startrow=SimpleDataFrame3.shape[0] + 13, startcol=0)

worksheet.write_string(SimpleDataFrame4.shape[0] + 16, 0, SimpleDataFrame5.name)
SimpleDataFrame5.to_excel(writer,sheet_name='Result',startrow=SimpleDataFrame4.shape[0] + 17, startcol=0)

worksheet.write_string(SimpleDataFrame5.shape[0] + 20, 0, SimpleDataFrame6.name)
SimpleDataFrame6.to_excel(writer,sheet_name='Result',startrow=SimpleDataFrame5.shape[0] + 21, startcol=0)

writer.save()
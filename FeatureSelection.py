from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from numpy import mean
from numpy import std
import pandas as pd
import re
import numpy as np
import lightgbm as lgb

from numpy.testing.tests.test_doctesting import check_empty_output
from dask.array.tests.test_gufunc import test_apply_gufunc_output_dtypes_string_many_outputs
from sklearn.feature_selection._from_model import SelectFromModel
from sklearn.ensemble._forest import RandomForestClassifier

#import excel file
file=pd.ExcelFile('cassandra.xlsx')
print ('Type of a file imported' ,type(file))
df1=file.parse('Sheet1')
print('Number of rows and columns in excel',df1.shape) 
#create a copy of data
df_max_column= df1.copy()                                                                                        
print(df_max_column.columns)
#apply normalisation (min-max)
for column in df_max_column.columns:
    df_max_column[column]= (df_max_column[column]- df_max_column[column].min())/(df_max_column[column].max()-df_max_column[column].min())
print(df_max_column)
Max_values=df_max_column.max(numeric_only= True)
print(Max_values)
Min_values=df_max_column.min(numeric_only=True)
print(Min_values)
'create Output '
y_output=df1['Output']
features1=df1.drop('Output', axis=1)
feature_list1=list(features1.columns)
features1=np.array(features1)
#apply scaler normalisation
scalar=StandardScaler()
scale1=scalar.fit_transform(features1)
print(scale1)
finalfeature1=pd.DataFrame(data=scale1,columns=feature_list1)
X=finalfeature1
y=y_output 
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.4, random_state=0)
'create a random forest classifier'
rf = SelectFromModel(RandomForestClassifier(n_estimators=1000))
rf.fit(X_train,y_train)
rf.get_support()
selected_feat= X_train.columns[(rf.get_support())]
print("Length of selected features: %d" %len(selected_feat))
print("Features from random Forest: %s" %selected_feat)
dfrandon_forest=pd.DataFrame(selected_feat)
dfrandon_forest.to_excel (r'C:\Users\Harguneet Kaur\eclipse-workspace\Optimization\random_forest.xlsx', index = False, header=True)



# Feature Extraction with RFE (Recursive Feature Elimination)
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# load data
y_output=df1['Output']
features1=df1.drop('Output', axis=1)
feature_list1=list(features1.columns)
features1=np.array(features1)
scalar=StandardScaler()
scale1=scalar.fit_transform(features1)
print(scale1)
finalfeature1=pd.DataFrame(data=scale1,columns=feature_list1)
X=finalfeature1
y=y_output 
#define RFE
rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=15)
# fit RFE
rfe.fit(X, y)
selected_features= X_train.columns[(rfe.get_support())]
# summarize all features
for i in range(X.shape[1]):
    print('Column: %d, Selected %s, Rank: %.3f' % (i, rfe.support_[i], rfe.ranking_[i]))
dfRFE=pd.DataFrame(selected_features)
dfRFE.to_excel (r'C:\Users\Harguneet Kaur\eclipse-workspace\Optimization\RFE_data.xlsx', index = False, header=True)


# automatically select the number of features for RFE
# create pipeline
rfea = RFECV(estimator=DecisionTreeClassifier())
model = DecisionTreeClassifier()
pipeline = Pipeline(steps=[('s',rfea),('m',model)])
# evaluate model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

# feature extraction with Logistic Regression
model = LogisticRegression(solver='lbfgs')
LogR = RFE(model, 10)
fit = LogR.fit(X, y)
selected_features_LR= fit.support_
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)
dfLR=pd.DataFrame(selected_features_LR)
dfLR.to_excel (r'C:\Users\Harguneet Kaur\eclipse-workspace\Optimization\LR_data.xlsx', index = False, header=True)


# Feature Importance with Extra Trees Classifier
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
# load data

# feature extraction
model = ExtraTreesClassifier(n_estimators=10)
FitExtraTree = model.fit(X, y)
print("Num Features for extra tree: %d" % FitExtraTree.n_features_)
print(model.feature_importances_)
fea_imp_ET = pd.DataFrame({'cols':X.columns, 'fea_imp':model.feature_importances_})
f=fea_imp_ET.loc[fea_imp_ET.fea_imp > 0].sort_values(by=['fea_imp'], ascending = False)
dfExtraTree=pd.DataFrame(f)
dfExtraTree.to_excel (r'C:\Users\Harguneet Kaur\eclipse-workspace\Optimization\ExtraTree_data.xlsx', index = False, header=True)

# Feature importance

#lightGBM model fit
gbm = lgb.LGBMRegressor()
GBC=gbm.fit(X, y)
gbm.booster_.feature_importance()

# importance of each attribute
fea_imp_ = pd.DataFrame({'cols':X.columns, 'fea_imp':gbm.feature_importances_})
f=fea_imp_.loc[fea_imp_.fea_imp > 0].sort_values(by=['fea_imp'], ascending = False)
print(f)
selected_features_lightGBM= f
dflightGBM=pd.DataFrame(selected_features_lightGBM)
dflightGBM.to_excel (r'C:\Users\Harguneet Kaur\eclipse-workspace\Optimization\GBM_data.xlsx', index = False, header=True)
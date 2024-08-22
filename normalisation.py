from sklearn.preprocessing import StandardScaler
from numpy import mean
from numpy import std
import pandas as pd
from pandas import DataFrame
import re
import numpy as np
from dask.dataframe.core import DataFrame
#import excel file
file=pd.ExcelFile('hadoop_hdfs.xlsx')
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
df_max_column.to_excel (r'C:\Users\Harguneet Kaur\eclipse-workspace\Optimization\min_max_normalised.xlsx', index = False, header=True) 

'create Output '
y_output=df1['Output']
features1=df1.drop('Output', axis=1)
feature_list1=list(features1.columns)
features1=np.array(features1)
#apply scaler normalisation
scalar=StandardScaler()
scale1=scalar.fit_transform(features1)
print(scale1)
dfscale=pd.DataFrame(scale1)
dfscale.to_excel (r'C:\Users\Harguneet Kaur\eclipse-workspace\Optimization\scale_data.xlsx', index = False, header=True) 
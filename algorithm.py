import pandas as pd
import re
file=pd.ExcelFile('HiveBugs.xlsx')
print ('Type of a file imported' ,type(file))
df1=file.parse('Sheet1')
print('Number of rows and columns in excel',df1.shape) 
list=['race', 'leak','memory','aging','overflow','deplet','Overflow','NPE', 'null pointer', 'Buffer exhausted', 'deadlock', 'flush', 'Leak','Memory','LEAK', 'MEMORY','OVERFLOW', 'null pointer exception' ]
df2=df1['Summary'].astype(str) + df1['Issue id'].astype(str) + df1['Issue key'].astype(str)
finallist=[]
'Coversion of dataframe into list'
listsum = df2.values.tolist()

'to check if there are unicode characters'
for j in listsum:
    print(j)
                
print('Number of keywords in a list', len(list))
print('Total Number of Bug Reports ', len(listsum))
'To check if keywords are present in bug reports'
for e in list:
    for j in listsum:
        if e in j: 
             '''print(j)'''
             finallist.append(j)
print('Number of Aging Related Bugs with duplicates', len(finallist))  
'below code is to remove duplicates from the list'
seen = set()
result = []
for item in finallist:
    if item not in seen:
        seen.add(item)
        result.append(item)
print('Number of Aging Related Bugs without duplicates',len(result))
'Coversion of list back into dataframe'
df = pd.DataFrame(result)
'Results exported to excel file'
df.to_excel (r'C:\Users\Harguneet Kaur\eclipse-workspace\Basics\export_dataframe.xlsx', index = False, header=True) 
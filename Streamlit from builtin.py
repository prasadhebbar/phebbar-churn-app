#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
import pickle


# In[2]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df_churn = pd.read_csv('telco_churn.csv')
print(df_churn.head())


# In[3]:


df_churn = df_churn[['gender', 'PaymentMethod', 'MonthlyCharges','tenure', 'Churn']].copy()
print(df_churn.head())


# In[4]:


df = df_churn.copy()
df.fillna(0, inplace=True)


# In[5]:


encode = ['gender','PaymentMethod']

for col in encode:

    dummy = pd.get_dummies(df[col], prefix=col)

    df = pd.concat([df,dummy], axis=1)

    del df[col]


# In[7]:


import numpy as np 

df['Churn'] = np.where(df['Churn']=='Yes', 1, 0)

# Now, let’s define our input and output :

X = df.drop('Churn', axis=1)

Y = df['Churn']


# In[8]:


clf = RandomForestClassifier()

clf.fit(X, Y)


# In[9]:


pickle.dump(clf, open('churn_clf.pkl', 'wb'))


# In[ ]:





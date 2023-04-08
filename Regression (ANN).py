#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('Crop_recommendation.csv')


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


x = data.iloc[:,3:-1].values
y = data.iloc[:,-1].values


# In[6]:


x


# In[7]:


y


# In[8]:


from sklearn.preprocessing import LabelEncoder


# In[9]:


le = LabelEncoder()
x[:,2] = le.fit_transform(x[:,2])


# In[10]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# In[11]:


ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1])])
x = np.array(ct.fit_transform(x))


# In[12]:


one_hot = pd.get_dummies(data['label'])
data = data.drop('label', axis=1)
data = pd.concat([data, one_hot], axis=1)


# In[13]:


X = data.drop(['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate', 'rice', 'watermelon'], axis=1)
y = data[['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate', 'rice', 'watermelon']]


# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[15]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[16]:


lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)
r2_lin = r2_score(y_test, y_pred_lin)
print(r2_lin)


# In[17]:


mlp_reg = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=500, activation='relu', solver='adam', random_state=42)
mlp_reg.fit(X_train, y_train)
y_pred_mlp = mlp_reg.predict(X_test)
r2_mlp = r2_score(y_test, y_pred_mlp)
print(r2_mlp)


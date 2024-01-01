#!/usr/bin/env python
# coding: utf-8

# # Problem Statement
# 
# The Big Mart Sales dataset on Kaggle aims to create a predictive model using machine learning techniques for forecasting product sales in the Big Mart retail chain. The goal is to predict continuous sales figures for various products across different stores based on historical sales data and relevant features. This model will aid Big Mart in optimizing inventory management, stock replenishment strategies, and store performance by providing accurate sales forecasts.

# # Library:-

# To initiate our analysis, we imported essential libraries for data manipulation and visualization. Subsequently, the Big Mart Sales dataset was loaded seamlessly into our environment using Pandas, a Python library for data manipulation.

# In[99]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# # Load DataSet:-

# In[100]:


df_train=pd.read_csv("Train.csv")
df_test=pd.read_csv("Test.csv")


# In[101]:


df_train


# In[102]:


df_test


# # Manally EDA

# In[103]:


df_train.shape


# In[104]:


df_test.shape


# In[105]:


df_train.isnull().sum()


# In[106]:


df_test.isnull().sum()


# In[107]:


df_train.info()


# In[108]:


df_test.info()


# In[109]:


df_train.describe()


# In[110]:


df_test.describe()


# In[111]:


for i in df_train.columns:
    print(i)
    print(df_train[i].value_counts())
    print("*********************\n")


# # As I seen- The dataset consists of both numerical and categorical variables

# # We have two column which have missing value 
#  -Item_Weight
#  
#  -Outlet_Size

# # Item_Weight is numerical column so we fill it with Mean Imputation
# 

# In[112]:


df_train['Item_Weight'].describe()


# In[113]:


df_train['Item_Weight'].fillna(df_train['Item_Weight'].mean(),inplace=True)
df_test['Item_Weight'].fillna(df_test['Item_Weight'].mean(),inplace=True)


# In[114]:


df_train.isnull().sum()


# In[115]:


df_train['Item_Weight'].describe()


# # Outlet_Size is catagorical column so we fill it with Mode Imputation

# In[116]:


df_train['Outlet_Size'].value_counts()


# In[117]:


df_train['Outlet_Size'].mode()


# In[118]:


df_train['Outlet_Size'].fillna(df_train['Outlet_Size'].mode()[0],inplace=True)
df_test['Outlet_Size'].fillna(df_test['Outlet_Size'].mode()[0],inplace=True)


# In[119]:


df_train.isnull().sum()


# In[120]:


df_test.isnull().sum()


# # Selecting features based on general requirements

# In[121]:


df_train.drop(['Item_Identifier','Outlet_Identifier'],axis=1,inplace=True)
df_test.drop(['Item_Identifier','Outlet_Identifier'],axis=1,inplace=True)


# In[122]:


df_train


# In[126]:


plt.figure(figsize=(6,6))
sns.displot(df_train['Item_Weight'])
plt.show()


# In[127]:


plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Establishment_Year', data=df_train)
plt.show()


# In[128]:


plt.figure(figsize=(6,6))
sns.countplot(x='Item_Fat_Content', data=df_train)
plt.show()


# # EDA using Pandas Profiling

# In[123]:


from pandas_profiling import ProfileReport


# In[124]:


profile = ProfileReport(df_train, title="Pandas Profiling Report")


# In[125]:


profile


# In[ ]:





# # Preprocessing Task before Model Building
# 

# In[131]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[132]:


df_train['Item_Fat_Content']= le.fit_transform(df_train['Item_Fat_Content'])
df_train['Item_Type']= le.fit_transform(df_train['Item_Type'])
df_train['Outlet_Size']= le.fit_transform(df_train['Outlet_Size'])
df_train['Outlet_Location_Type']= le.fit_transform(df_train['Outlet_Location_Type'])
df_train['Outlet_Type']= le.fit_transform(df_train['Outlet_Type'])


# In[133]:


df_train


# In[149]:


### check colleration for all columns
train_corr = df_train.corr()
mask = np.triu(np.ones_like(train_corr,dtype=bool))

plt.figure(figsize=(13,10))
sns.heatmap(train_corr, cmap='RdYlGn_r', mask=mask , annot=True)
plt.xticks(rotation=65)
plt.show()


# # 2) Splitting our data into train and test
# 

# In[134]:


X=df_train.drop('Item_Outlet_Sales',axis=1)
Y=df_train['Item_Outlet_Sales']


# In[135]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=101,test_size=0.2) 


# In[136]:


X.describe()


# # Data Standardization
# 

# In[137]:


from sklearn.preprocessing import StandardScaler
sc= StandardScaler()


# In[138]:


X_train_std= sc.fit_transform(X_train)


# In[139]:


X_test_std= sc.transform(X_test)


# In[140]:


X_train_std


# In[141]:


X_test_std


# In[142]:


Y_train


# In[143]:


Y_test


# # Model Building

# In[145]:



from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[146]:


models = [LinearRegression, Lasso, Ridge, SVR, DecisionTreeRegressor, RandomForestRegressor]
mae_scores = []
mse_scores = []
rmse_scores = []
r2_scores = []

for model in models:
    regressor = model().fit(X_train, Y_train)
    Y_pred = regressor.predict(X_test)
    
    mae_scores.append(mean_absolute_error(Y_test, Y_pred))
    mse_scores.append(mean_squared_error(Y_test, Y_pred))
    rmse_scores.append(mean_squared_error(Y_test, Y_pred, squared=False))
    r2_scores.append(r2_score(Y_test, Y_pred))


# In[147]:


regression_metrics_df = pd.DataFrame({
    "Model": ["Linear Regression", "Lasso", "Ridge", "SVR", "Decision Tree Regressor", "Random Forest Regressor"],
    "Mean Absolute Error": mae_scores,
    "Mean Squared Error": mse_scores,
    "Root Mean Squared Error": rmse_scores,
    "R-squared (R2)": r2_scores
})

regression_metrics_df.set_index('Model', inplace=True)
regression_metrics_df


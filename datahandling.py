#!/usr/bin/env python
# coding: utf-8

# In[88]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
plt.rcParams['figure.figsize']=(5,5)


# In[2]:


data=pd.read_csv(r'C:\Users\Tanisha\OneDrive\Documents\RealTime_Project\healthcare-dataset-stroke-data.csv')


# In[3]:


data


# # Exploratory data Analysis

# In[4]:


data.info()


# In[5]:


data.isnull().sum()


# # Lets fill Null Values

# In[6]:


data['bmi'].value_counts()


# In[7]:


data['bmi'].describe()


# In[8]:


data['bmi'].fillna(data['bmi'].mean(),inplace=True)#filling null values


# In[9]:


data['bmi'].describe()


# In[10]:


data.isnull().sum()


# In[11]:


data.drop('id',axis=1,inplace=True)


# In[12]:


data


# # Outlier Removation
# 

# In[13]:


#Outlier removation: Remove those values which are very different from other 
data.plot(kind='box')#checking outliers.
plt.show()


# In[14]:


data['avg_glucose_level'].describe()


# # Label Encoding
# 

# In[15]:


data.head()


# In[16]:


from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()


# In[17]:


gender=enc.fit_transform(data['gender'])


# In[18]:


smoking_status=enc.fit_transform(data['smoking_status'])


# In[19]:


work_type=enc.fit_transform(data['work_type'])
Residence_type=enc.fit_transform(data['Residence_type'])
ever_married=enc.fit_transform(data['ever_married'])
gender=enc.fit_transform(data['gender'])


# In[20]:


Residence_type #run all three


# In[21]:


data['work_type']=work_type


# In[22]:


data['ever_married']=ever_married
data['Residence_type']=Residence_type
data['smoking_status']=smoking_status
data['gender']=gender


# In[23]:


data


# In[24]:


data.info()


# # Splitting the data for train and test

# X---train_X,test_X 80/20
# Y---train_Y,test_Y 

# In[25]:


X=data.drop('stroke',axis=1)


# In[26]:


X.head()


# In[27]:


Y=data['stroke']


# In[28]:


Y


# In[29]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=101)


# In[30]:


X_train


# In[31]:


X_test


# In[32]:


Y_train


# In[33]:


Y_test


# # Normalize
# 

# In[34]:


data.describe()


# In[ ]:





# In[35]:


from sklearn.preprocessing import StandardScaler
std=StandardScaler()


# In[36]:


X_train_std=std.fit_transform(X_train)
X_test_std=std.transform(X_test)


# In[37]:


X_train_std


# # X_test_std
# 

# # training

# In[ ]:





# # Desicion Tree

# In[38]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()


# In[39]:


dt.fit(X_train_std,Y_train)


# In[40]:


dt.feature_importances_


# In[41]:


X_train.columns


# In[42]:


Y_pred=dt.predict(X_test_std)


# In[43]:


Y_test


# In[44]:


X_test


# In[45]:


from sklearn.metrics import accuracy_score


# In[46]:


ac_dt=accuracy_score(Y_test,Y_pred)


# In[47]:


ac_dt


# # Logistic Regression

# In[48]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()


# In[49]:


lr.fit(X_train_std,Y_train)


# In[50]:


Y_pred=lr.predict(X_test_std)


# In[51]:


ac_lr=accuracy_score(Y_test,Y_pred)


# In[52]:


ac_lr


# # KNN

# In[53]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()


# In[54]:


knn.fit(X_train_std,Y_train)


# In[55]:


Y_pred=knn.predict(X_test_std)


# In[58]:


ac_knn=accuracy_score(Y_test,Y_pred)


# In[59]:


ac_knn


# In[57]:


ac_lr


# # Random Forest

# In[60]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()


# In[66]:


rf.fit(X_train,Y_train)


# In[67]:


Y_pred=rf.predict(X_test)


# In[68]:


ac_rf=accuracy_score(Y_test,Y_pred)


# In[69]:


ac_rf


# In[70]:


ac_knn


# In[71]:


ac_dt


# In[72]:


ac_lr


# # SVM

# In[73]:


from sklearn.svm import SVC


# In[74]:


sv=SVC()


# In[76]:


sv.fit(X_train_std,Y_train)


# In[80]:


Y_pred=sv.predict(X_test_std)


# In[81]:


ac_sv=accuracy_score(Y_test,Y_pred)


# In[82]:


ac_sv


# In[79]:


ac_lr


# In[ ]:





# In[92]:


plt.bar(['Decision tree','Logistic','KNN','Random Forest','SVM'],[ac_dt,ac_lr,ac_knn,ac_rf,ac_sv])
plt.xlabel("Algorithms")
plt.ylabel("Accuracy")
plt.show()


# In[ ]:


import pickle
model=open(b"","wb")


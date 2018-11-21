#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


dataset=pd.read_csv('BlackFriday.csv').sample(100000, random_state=44)


# In[4]:


dataset.head()


# In[5]:


dataset.shape


# In[6]:


dataset['Gender'].unique()


# In[7]:


dataset['Gender'].value_counts().plot.bar()


# In[9]:


dataset['Age'].value_counts().sort_values(ascending=False).plot.bar()


# In[8]:


dataset['Occupation'].value_counts().plot.bar()


# In[11]:


total_occ=len(dataset)
occ_df = pd.Series(dataset['Occupation'].value_counts() / total_occ).reset_index()
occ_df.columns = ['Occupation', 'Occupation_percent']
occ_df


# In[10]:


dataset.groupby(['Occupation'])['Purchase'].mean().reset_index()


# In[12]:


occ_df = occ_df.merge(
    dataset.groupby(['Occupation'])['Purchase'].mean().reset_index(), on='Occupation', how='left')

occ_df


# In[13]:


fig, ax = plt.subplots(figsize=(8, 4))
plt.xticks(occ_df.index, occ_df['Occupation'], rotation=0)

ax2 = ax.twinx()
ax.bar(occ_df.index, occ_df["Occupation_percent"], color='lightgrey')
ax2.plot(occ_df.index, occ_df["Purchase"], color='green', label='Seconds')
ax.set_ylabel('percentage of cars per category')
ax2.set_ylabel('Seconds')


# In[14]:


occ_df = pd.Series(dataset['Occupation'].value_counts() / total_occ)
occ_df.sort_values(ascending=False)
occ_df




occ_df[occ_df>=0.05].index


# In[15]:


grouping_dict={
    k:('rare' if k not in occ_df[occ_df>=0.05].index else k)
    for k in occ_df.index
}

grouping_dict


# In[18]:


dataset['Occupation_grouped']=dataset['Occupation'].map(grouping_dict)

dataset[['Occupation','Occupation_grouped']].head(30)
                                                 


# In[19]:


dataset['Occupation_grouped'].value_counts().plot.bar()


# In[20]:


dataset['City_Category'].unique()


# In[21]:


dataset['City_Category'].value_counts().plot.bar()


# In[22]:


dataset.head()


# In[23]:


dataset.Marital_Status.value_counts().plot.bar()


# In[24]:


from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()



# In[25]:


dataset['Gender']=label.fit_transform(dataset['Gender'])
dataset['Age']=label.fit_transform(dataset['Age'])

dataset['Marital_Status']=label.fit_transform(dataset['Marital_Status'])
dataset['City_Category']=label.fit_transform(dataset['City_Category'])


# In[26]:


dataset.head()


# In[27]:


dataset.Product_Category_1.unique()


# In[28]:


dataset.Product_Category_1.value_counts().plot.bar()


# In[31]:


total_pdt=len(dataset)
pdt_df = pd.Series(dataset['Product_Category_1'].value_counts() / total_pdt).reset_index()
pdt_df.columns = ['Product_Category_1', 'Product_Category_1_precentage']
pdt_df


# In[32]:


pdt_df = pdt_df.merge(
    dataset.groupby(['Product_Category_1'])['Purchase'].mean().reset_index(), on='Product_Category_1', how='left')

pdt_df


# In[33]:


fig, ax = plt.subplots(figsize=(8, 4))
plt.xticks(pdt_df.index, pdt_df['Product_Category_1'], rotation=0)

ax2 = ax.twinx()
ax.bar(pdt_df.index, pdt_df["Product_Category_1_precentage"], color='lightgrey')
ax2.plot(pdt_df.index, pdt_df["Purchase"], color='green', label='Seconds')
ax.set_ylabel('percentage of cars per category')
ax2.set_ylabel('Seconds')


# In[34]:


pdt_df = pd.Series(dataset['Product_Category_1'].value_counts() / total_occ)
pdt_df.sort_values(ascending=False)
pdt_df




pdt_df[pdt_df>=0.20].index


# In[35]:


grouping_dict_pdt={
    k:('rare' if k not in pdt_df[pdt_df>=0.20].index else k)
    for k in pdt_df.index
}

grouping_dict_pdt


# In[36]:


dataset['Product_grouped_1']=dataset['Product_Category_1'].map(grouping_dict_pdt)

dataset[['Product_Category_1','Product_grouped_1']].head(30)


# In[37]:


dataset.Product_grouped_1.value_counts().plot.bar()


# In[38]:


dataset['Product_grouped_1']=dataset['Product_grouped_1'].astype(str)


# In[39]:


dataset['Product_grouped_1']=label.fit_transform(dataset['Product_grouped_1'])


# In[40]:


dataset.info()

dataset['Occupation_grouped']=dataset['Occupation_grouped'].astype(str)


# In[41]:


dataset['Occupation_grouped']=label.fit_transform(dataset['Occupation_grouped'])


# In[42]:


X=dataset[['Gender','Age','Occupation','City_Category','Marital_Status','Occupation_grouped','Product_grouped_1']]


# In[43]:


X.head()


# In[44]:


y=dataset['Purchase']


# In[45]:


from sklearn.model_selection import train_test_split


# In[46]:


X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=44,test_size=0.3)


# In[47]:


X_train.shape


# In[48]:


X_test.shape


# In[49]:


from sklearn.linear_model import LogisticRegression


logit = LogisticRegression(random_state=44)



# In[ ]:


logit.fit(X_train, y_train)


# In[ ]:


pred = logit.predict_proba(X_test)


# In[1]:



print('LogReg Accuracy: {}'.format(logit.score(X_test, y_test)))
print('LogReg roc-auc: {}'.format(roc_auc_score(y_test, pred[:, 1])))


# In[ ]:





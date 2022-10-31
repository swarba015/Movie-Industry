#!/usr/bin/env python
# coding: utf-8

# In[19]:


#Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib 
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8) #adjusts the configuration of the plot we will create



# In[23]:


#Import and read data
df = pd.read_csv("C:\\Users\\Towfiq\\Documents\\movies.csv")
df.head()


# In[24]:


#Shape of data set
df.shape


# In[25]:


# Data Types of columns

print(df.dtypes)


# In[21]:


#Finding missing data
for col in df.columns:
    missing_data = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round(missing_data*100)))


# In[26]:


#Cleaning data
df2x = df.dropna(axis=0, subset=['rating','budget','gross'])
df2x.info()


# In[33]:


#Changing data type of votes
df2x = df2x.astype({"votes" :np.dtype("int64"), "budget" : np.dtype("int64"), "gross" : np.dtype("int64")})
df2x.info()


# In[36]:


df2x["company"] = df2x["company"].drop_duplicates().sort_values()
df2x.info()


# In[42]:


#Ordering data to understand better

df2x_sorting_gross_values = df2x.sort_values(by=['gross'], inplace=False, ascending=False)
df2x_sorting_gross_values.head()


# In[43]:


# scatterplot with budget vs gross 
plt.scatter(x=df2x['budget'], y=df2x['gross'])
plt.title('Budget vs Gross Revenue')
plt.ylim([0,2*10**9])
plt.xlabel('Gross Revenue')
plt.ylabel('Budget for Films')


# In[47]:


#budget vs gross (regplot)
sns.regplot(x='budget', y='gross', data = df2x, scatter_kws={'color':'red'}, line_kws={'color':'blue'})
plt.ylim([0,2*10**9])


# In[49]:


# Tho correlation table looks as follows.
df.corr(method = 'pearson')


# In[51]:


# plotting the Heatmap

corr_mat = df.corr(method='pearson')
sns.heatmap(corr_mat, annot=True)
plt.title('Correlation Metric for Numberic features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')


# In[52]:


#Numerization of Object Data Types
df_numerized = df2x
for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype == 'object'):
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes
df_numerized.head()


# In[53]:


correlation_matrix = df_numerized.corr(method='pearson')
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Metric for Numberic features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')


# In[ ]:





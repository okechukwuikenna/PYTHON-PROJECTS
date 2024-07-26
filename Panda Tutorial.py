#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Importing Pandas
import pandas as pd
import warnings

# Ignoring warnings
warnings.filterwarnings("ignore")

# Creating a sample DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Edward'],
    'Age': [24, 27, 22, 32, 29],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
    'Salary': [70000, 80000, 65000, 90000, 75000]
}

df = pd.DataFrame(data)
print(df)


# In[4]:


df.head() 


# In[8]:


df.info() 


# In[9]:


df.describe()


# In[10]:


df['Experience'] = [2, 5, 1, 7, 3] 
df


# In[12]:


mask = df['Age'] > 25
df[mask] 


# In[13]:


df['Cumulative Salary'] = df['Salary'].cumsum() # Adds a column with the cumulative sum of 'Salary'
df


# In[14]:


df['Rolling Mean Salary'] = df['Salary'].rolling(window=2).mean() # Adds a column with the rolling mean of 'Salary' over a window of 2 rows
df


# In[18]:


# Applying multiple aggregate functions
grouped_agg = df.groupby('City').agg({'Age': ['mean', 'max'], 'Salary': ['sum', 'mean']}) # Groups by 'Location' and applies multiple aggr
grouped_agg


# In[19]:


# Grouping by a column and applying a custom function
def custom_function(x):
 return x.max() - x.min()
grouped_custom = df.groupby('City')['Salary'].apply(custom_function) # Groups by 'Location' and applies a custom function to 'Salary'
grouped_custom


# In[20]:


# Filtering using a custom function
def filter_function(x):
 return x['Age'].mean() > 25
filtered = df.groupby('City').filter(filter_function) # Filters groups where the mean 'Age' is greater than 25
filtered


# In[21]:


# Filtering with the where() method
df.where(df['Salary'] > 75000, other=0) # Replaces values where 'Salary' is not greater than 75000 with 0


# In[22]:


# Highlighting maximum values in a DataFrame
df.style.highlight_max(axis=0) # Highlights the maximum values in each column


# In[23]:


# Applying a gradient based on values
df.style.background_gradient(cmap='viridis') # Applies a color gradient based on values


# In[24]:


# Applying a custom function for styling
def color_negative_red(val):
 color = 'red' if val < 75000 else 'black'
 return 'color: {}'.format(color)
df.style.applymap(color_negative_red, subset=['Salary']) # Colors 'Salary' values red if they are less than 75000


# In[ ]:





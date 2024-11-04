#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

# Load the data
df = pd.read_csv('polluting.csv')

# Display the first few rows of the dataframe
print(df.head())

# Display data types and check for missing values
print(df.info())



# In[5]:


# Convert 'year' to datetime
df['Year'] = pd.to_datetime(df['Year'], format='%Y')

# Check for missing values and drop them if necessary
df = df.dropna()


# In[6]:


# Set the year as the index
df.set_index('Year', inplace=True)


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for seaborn
sns.set(style="whitegrid")

# Plotting the time series
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Overall'], label='Overall', color='blue')
plt.fill_between(df.index, df['Overall Lower Limit'], df['Overall Upper Limit'], color='lightblue', alpha=0.5, label='Limits')
plt.title('Time Series of Pollution Data')
plt.xlabel('Year')
plt.ylabel('Overall Pollution')
plt.legend()
plt.show()


# In[9]:


# Calculate the rolling average
df['rolling_mean'] = df['Overall'].rolling(window=5).mean()

# Plot the rolling mean
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Overall'], label='Overall', color='blue')
plt.plot(df.index, df['rolling_mean'], label='Rolling Mean (5-year)', color='orange', linestyle='--')
plt.title('Overall Pollution with Rolling Mean')
plt.xlabel('Year')
plt.ylabel('Overall Pollution')
plt.legend()
plt.show()


# In[10]:


from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose the time series
decomposition = seasonal_decompose(df['Overall'], model='additive')
decomposition.plot()
plt.show()


# In[11]:


from statsmodels.tsa.arima.model import ARIMA

# Fit the ARIMA model (choose order (p, d, q) based on ACF and PACF plots)
model = ARIMA(df['Overall'], order=(1, 1, 1))
model_fit = model.fit()

# Print the summary of the model
print(model_fit.summary())

# Forecasting the next 5 years
forecast = model_fit.forecast(steps=5)
print(forecast)


# In[12]:


from statsmodels.tsa.stattools import adfuller

# Perform the ADF test
result = adfuller(df['Overall'])

# Extract and display the results
adf_statistic = result[0]
p_value = result[1]
used_lag = result[2]
num_obs = result[3]
critical_values = result[4]

print('ADF Statistic:', adf_statistic)
print('p-value:', p_value)
print('Used Lag:', used_lag)
print('Number of Observations:', num_obs)
print('Critical Values:')
for key, value in critical_values.items():
    print(f'   {key}: {value}')


# In[ ]:





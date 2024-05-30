#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[69]:


#Three lines to make our compiler able to draw:
import sys
import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

full_health_data = pd.read_csv("dataH.csv", header=0, sep=",")

x = full_health_data["Average_Pulse"]
y = full_health_data["Calories_Burnage"]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
 return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.ylim(ymin=0, ymax=2000)
plt.xlim(xmin=0, xmax=200)
plt.xlabel("Average_Pulse")
plt.ylabel ("Calories_Burnage")
plt.show()


# In[15]:


import pandas as pd

d = {'col1': [1, 2, 3, 4, 7], 'col2': [4, 5, 6, 9, 5], 'col3': [7, 8, 12, 1, 11]}

df = pd.DataFrame(data=d)

print(df)


# In[16]:


count_column = df.shape[1]
print(count_column)


# In[17]:


Average_pulse_max = max(80, 85, 90, 95, 100, 105, 110, 115, 120, 125)

print (Average_pulse_max)


# In[18]:


# Extract and read data with panda:
import pandas as pd

health_data = pd.read_csv("dataH.csv", header=0, sep=",")

print(health_data)


# In[19]:


import pandas as pd

health_data = pd.read_csv("dataH.csv", header=0, sep=",")

print(health_data.head())


# In[20]:


#Data Cleaning:
health_data.dropna(axis=0,inplace=True)

print(health_data)


# In[22]:


print(health_data.info())


# In[32]:


health_data["Average_Pulse"] = health_data['Average_Pulse'].astype(float)
health_data["Maxpulse"] = health_data["Maxpulse"].astype(float)

print (health_data.info())


# In[33]:


print(health_data.describe())


# In[34]:


#plotting linear functions:
import matplotlib.pyplot as plt

health_data.plot(x ='Average_Pulse', y='Calories_Burnage', kind='line'),
plt.ylim(ymin=0)
plt.xlim(xmin=0)

plt.show()


# In[35]:


def slope(x1, y1, x2, y2):
  s = (y2-y1)/(x2-x1)
  return s

print (slope(80,240,90,260))


# In[36]:


import pandas as pd
import numpy as np

health_data = pd.read_csv("dataH.csv", header=0, sep=",")

x = health_data["Average_Pulse"]
y = health_data["Calories_Burnage"]
slope_intercept = np.polyfit(x,y,1)

print(slope_intercept)


# In[37]:


import matplotlib.pyplot as plt

health_data.plot(x ='Average_Pulse', y='Calories_Burnage', kind='line'),
plt.ylim(ymin=0, ymax=400)
plt.xlim(xmin=0, xmax=150)

plt.show()


# In[40]:


# If you are using Jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

health_data.plot(x ='Average_Pulse', y='Calories_Burnage', kind='line'),
plt.ylim(ymin=0, ymax=400)
plt.xlim(xmin=0, xmax=150)
health_data = pd.read_csv("dataH.csv", header=0, sep=",")


linkage_data = linkage(data, method='ward', metric='euclidean')
dendrogram(linkage_data)

plt.show()


# In[39]:


# If you are using Jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
data = list(zip(x, y))

linkage_data = linkage(data, method='ward', metric='euclidean')
dendrogram(linkage_data)

plt.show()


# In[41]:


#Descriptive Statistics:
print (full_health_data.describe())


# In[44]:


#percentiles
import numpy as np

Maxpulse= full_health_data["Maxpulse"]
percentile10 = np.percentile(Maxpulse, 10)
print(percentile10)


# In[45]:


#standard deviation
import numpy as np

std = np.std(full_health_data)
print(std)


# In[46]:


import numpy as np

cv = np.std(full_health_data) / np.mean(full_health_data)
print(cv)


# In[49]:


#variance
import numpy as np

var_full = np.var(full_health_data)
print(var_full)


# In[50]:


import numpy as np

var = np.var(health_data)
print(var)


# In[51]:


#correlation Coefficient
import matplotlib.pyplot as plt

health_data.plot(x ='Average_Pulse', y='Calories_Burnage', kind='scatter')
plt.show()


# In[52]:


import pandas as pd
import matplotlib.pyplot as plt

negative_corr = {'Hours_Work_Before_Training': [10,9,8,7,6,5,4,3,2,1],
'Calorie_Burnage': [220,240,260,280,300,320,340,360,380,400]}
negative_corr = pd.DataFrame(data=negative_corr)

negative_corr.plot(x ='Hours_Work_Before_Training', y='Calorie_Burnage', kind='scatter')
plt.show()


# In[53]:


import matplotlib.pyplot as plt

full_health_data.plot(x ='Duration', y='Maxpulse', kind='scatter')
plt.show()


# In[54]:


#correlation Matrix:
Corr_Matrix = round(full_health_data.corr(),2)
print(Corr_Matrix)


# In[55]:


#creating a heatmap with seaborn:
import matplotlib.pyplot as plt
import seaborn as sns

correlation_full_health = full_health_data.corr()

axis_corr = sns.heatmap(
correlation_full_health,
vmin=-1, vmax=1, center=0,
cmap=sns.diverging_palette(50, 500, n=500),
square=True
)

plt.show()


# In[57]:


#correlation vs casuality
import pandas as pd
import matplotlib.pyplot as plt

Drowning_Accident = [20,40,60,80,100,120,140,160,180,200]
Ice_Cream_Sale = [20,40,60,80,100,120,140,160,180,200]
Drowning = {"Drowning_Accident": [20,40,60,80,100,120,140,160,180,200],
"Ice_Cream_Sale": [20,40,60,80,100,120,140,160,180,200]}
Drowning = pd.DataFrame(data=Drowning)

Drowning.plot(x="Ice_Cream_Sale", y="Drowning_Accident", kind="bar")
plt.show()

correlation_beach = Drowning.corr()
print(correlation_beach)


# In[59]:


#linear regression
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

full_health_data = pd.read_csv("dataH.csv", header=0, sep=",")

x = full_health_data["Average_Pulse"]
y = full_health_data ["Calories_Burnage"]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
 return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, slope * x + intercept)
plt.ylim(ymin=0, ymax=2000)
plt.xlim(xmin=0, xmax=200)
plt.xlabel("Average_Pulse")
plt.ylabel ("Calories_Burnage")
plt.show()


# In[60]:


#Regression Table:
import pandas as pd
import statsmodels.formula.api as smf

full_health_data = pd.read_csv("dataH.csv", header=0, sep=",")

model = smf.ols('Calories_Burnage ~ Average_Pulse', data = full_health_data)
results = model.fit()
print(results.summary())


# In[62]:


# Linear Regression in Python:
def Predict_Calories_Burnage(Average_Pulse):
 return(0.3296*Average_Pulse + 346.8662)

print(Predict_Calories_Burnage(120))
print(Predict_Calories_Burnage(130))
print(Predict_Calories_Burnage(150))
print(Predict_Calories_Burnage(180))


# In[64]:


# Visual Examples of a High R-Squared 
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

full_health_data = pd.read_csv("dataH.csv", header=0, sep=",")

x = full_health_data["Duration"]
y = full_health_data ["Calories_Burnage"]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
 return slope * x + intercept

mymodel = list(map(myfunc, x))

print(mymodel)

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.ylim(ymin=0, ymax=2000)
plt.xlim(xmin=0, xmax=200)
plt.xlabel("Duration")
plt.ylabel ("Calories_Burnage")

plt.show()


# In[65]:


#case: Use Duration + Average_Pulse to Predict Calorie_Burnage
import pandas as pd
import statsmodels.formula.api as smf

full_health_data = pd.read_csv("dataH.csv", header=0, sep=",")

model = smf.ols('Calories_Burnage ~ Average_Pulse + Duration', data = full_health_data)
results = model.fit()
print(results.summary())


# In[66]:


def Predict_Calories_Burnage(Average_Pulse, Duration):
 return(3.1695*Average_Pulse + 5.8434 * Duration - 334.5194)

print(Predict_Calories_Burnage(110,60))
print(Predict_Calories_Burnage(140,45))
print(Predict_Calories_Burnage(175,20))


# In[ ]:






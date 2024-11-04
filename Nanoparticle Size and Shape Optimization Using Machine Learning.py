#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# TensorFlow and Keras for deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# In[2]:


# Simulating nanoparticle data (size, shape, surface area, and performance)
np.random.seed(42)

data_size = 1000  # Number of samples

# Simulated dataset for nanoparticle properties: size (nm), shape factor (arbitrary), surface area (m²/g)
size = np.random.uniform(10, 100, data_size)  # Size in nm
shape_factor = np.random.uniform(1, 5, data_size)  # Shape factor (arbitrary)
surface_area = np.random.uniform(50, 300, data_size)  # Surface area (m²/g)

# Performance metric based on nanoparticle properties
performance = 3 * size - 2 * shape_factor + 1.5 * surface_area + np.random.normal(0, 10, data_size)

# Create a DataFrame
df = pd.DataFrame({
    'Size (nm)': size,
    'Shape Factor': shape_factor,
    'Surface Area (m²/g)': surface_area,
    'Performance': performance
})

# Show the first few rows of the data
df.head()


# In[3]:


# Features (X) and target (y)
X = df[['Size (nm)', 'Shape Factor', 'Surface Area (m²/g)']]
y = df['Performance']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data (deep learning models typically perform better with standardized data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[4]:


# Define the deep learning model
model = Sequential()

# Input layer with 3 features (size, shape factor, surface area), hidden layers with ReLU activation
model.add(Dense(64, input_dim=3, activation='relu'))  # 64 neurons in the first hidden layer
model.add(Dense(32, activation='relu'))  # 32 neurons in the second hidden layer
model.add(Dense(16, activation='relu'))  # 16 neurons in the third hidden layer

# Output layer for regression (predicting performance)
model.add(Dense(1))

# Compile the model (using Mean Squared Error loss for regression)
model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['mean_squared_error'])

# Print the model summary
model.summary()


# In[5]:


# Train the model
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100, batch_size=32)

# Plot the training and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()


# In[10]:


# Evaluate the model on the test set
test_loss, test_mse = model.evaluate(X_test_scaled, y_test)
print(f'Test Mean Squared Error: {test_mse}')

# Predict the performance for the test set
y_pred = model.predict(X_test_scaled)

# Plot actual vs predicted performance
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Performance')
plt.ylabel('Predicted Performance')
plt.title('Actual vs Predicted Nanoparticle Performance')
plt.show()


# In[7]:


# Example: Predicting performance for a new nanoparticle configuration
new_nanoparticle = np.array([[50, 3, 200]])  # Size=50nm, Shape Factor=3, Surface Area=200 m²/g
new_nanoparticle_scaled = scaler.transform(new_nanoparticle)

predicted_performance = model.predict(new_nanoparticle_scaled)
print(f'Predicted Performance for new nanoparticle: {predicted_performance[0][0]}')


# In[ ]:





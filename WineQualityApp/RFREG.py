#!/usr/bin/env python
# coding: utf-8

# ## LOADING the DATA

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


df = pd.read_csv("clean_wine.csv")
df.info()


# ## Preparing the Data

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


# In[ ]:


X_test.shape , y_test.shape


# ## Logistic Regression 

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=30, random_state=42)
model.fit(X_train, y_train)


# In[ ]:


from sklearn.metrics import classification_report

# Predictions
y_pred = model.predict(X_test)

# Convert regression predictions → classes for F1 evaluation
y_pred_rounded = np.rint(y_pred).astype(int)
y_pred_rounded = np.clip(y_pred_rounded, y_test.min(), y_test.max())

print("Train R²:", model.score(X_train, y_train))
print("Test  R² :", model.score(X_test, y_test))

print("\nClassification Report (using rounded predictions):")
print(classification_report(y_test, y_pred_rounded, digits=3))

import pickle

with open("wine_rf_model.pkl", "wb") as file:
    pickle.dump(model, file)
    

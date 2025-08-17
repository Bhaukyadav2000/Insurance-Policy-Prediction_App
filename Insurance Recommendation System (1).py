#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


#Load the dataset
Insu_data = pd.read_csv("insurance_dataset_with_calc_risk.csv")


# In[4]:


Insu_data


# In[5]:


Insu_data.describe()


# In[6]:


# After describe we can conclude that there is no such ouliers in the data to remove, for Income we can treat a
# value as outlier as it can vary person to person.


# In[7]:


#Checking for NULL values
Insu_data.isnull().sum()


# In[8]:


X = Insu_data.drop("Recommended_Policy", axis=1)
y = Insu_data["Recommended_Policy"]


# In[9]:


# No Using label encoder to encode the values (By using this we can make the values on a same level like income
# income values are way higher than the other values so it may overpower the data, so we use label encoder)
# but here we only encoder the target column as the given values are not in numerical format.
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# In[10]:


# Train the Model on that data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)


# In[11]:


# No we are using streamlit to make a front-end to get the inputs from users.
st.title("🛡 Insurance Policy Recommendation System")
st.write("Fill in your details and get the **Top 3** matching insurance policies.")


# In[12]:


# Now we fix the input form of data to avoid any kind of errors
age = st.number_input("Age", min_value=18, max_value=100, value=35)
income = st.number_input("Annual Income (in USD)", min_value=1000, max_value=200000, value=50000)
family_size = st.number_input("Family Size", min_value=1, max_value=10, value=1)
existing_policies = st.number_input("No of Existing policies", min_value=0, max_value=10, value=1)
claims = st.number_input("Number of Claims in last year", min_value=0, max_value=10, value=1)
risk_score = st.slider("Risk Score (1 - Low, 10 - High)", min_value=1, max_value=10, value=5)
preferred_coverage = st.number_input("Preferred Coverage Value", min_value=100000, max_value=10000000, value=500000)


# In[13]:


if st.button("Get Recommendation"):
    new_customer = pd.DataFrame({
        "Age": [age],
        "Income": [income],
        "Family_Size": [family_size],              
        "Existing_Policies": [existing_policies],  
        "Claims": [claims],
        "Risk_Score": [risk_score],
        "Preferred_Coverage": [preferred_coverage]
    })

    prediction = model.predict(new_customer)[0]
    probs = model.predict_proba(new_customer)[0]

    st.success(f"Recommended Policy: **{prediction}**")
    st.write("Prediction Probabilities:")
    st.write({model.classes_[i]: round(probs[i]*100, 2) for i in range(len(model.classes_))})


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, recall_score, precision_score
import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

# In[10]:


df = pd.read_csv('C:\\Users\\nagak\Hackathon 2025\\Customer_Dataset.csv')

# In[21]:


import pandas as pd

# Assuming 'df' is your DataFrame
max_user_id = df['cust_id'].max()
print(f"Maximum user ID: {max_user_id}")
print(f"Number of unique users: {df['cust_id'].nunique()}")

# In[1]:


import pandas as pd
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares

# In[14]:


# Map customer & product IDs to sequential index values
customer_mapping = {id: i for i, id in enumerate(df["cust_id"].unique())}
product_mapping = {id: i for i, id in enumerate(df["item_id"].unique())}

df["customer_index"] = df["cust_id"].map(customer_mapping)
df["product_index"] = df["item_id"].map(product_mapping)

# In[15]:


# Create sparse matrix
customer_product_matrix = sparse.csr_matrix(
    (df["order_id"], (df["customer_index"], df["product_index"]))
)

# Train ALS model
model = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=20)
model.fit(customer_product_matrix)

# Ensure 'models/' directory exists inside your project folder
save_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(save_dir, exist_ok=True)

# Define correct path
correct_path = os.path.join(save_dir, "als_model.pkl")

# Save model
with open(correct_path, "wb") as file:
    pickle.dump(model, file)

print(f"Model saved successfully at {correct_path}!")

print("Sample customer IDs in mapping:", list(customer_mapping.keys())[:10])

# In[32]:


cust_id = 60124  # Replace with an actual customer ID from your dataset

if cust_id in customer_mapping:
    customer_index = customer_mapping[cust_id]
    recommendations = model.recommend(customer_index, customer_product_matrix[customer_index], N=5)

    # Debugging: Print the full recommendation output
    print("Raw Recommendations:", recommendations)

    # Extract recommended product indices and scores
    recommended_indices, scores = recommendations  # Correctly unpack the tuple

    # Convert indices back to actual product IDs
    recommended_product_ids = [list(product_mapping.keys())[i] for i in recommended_indices]

    print("Recommended Products:", recommended_product_ids)
else:
    print(f"Customer ID {customer_id} is not in the dataset. Try a different one.")

# In[ ]:

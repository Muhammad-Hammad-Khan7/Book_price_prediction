#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

# In[ ]:


df=pd.read_csv('/content/Books.csv')

# In[ ]:


df


# In[ ]:



y = df['premium']

# In[ ]:


y

# In[ ]:


x=df.drop(columns='premium')

# In[ ]:


x

# In[ ]:


from sklearn.model_selection import train_test_split

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# In[ ]:


x_train
x_test

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
regressor_rf = RandomForestRegressor()
regressor_rf.fit(x_train, y_train)

# In[ ]:


y_pred = regressor_rf.predict(x_test)

# In[ ]:


y_pred

# In[ ]:


import seaborn as sns
from sklearn.metrics import  confusion_matrix

# In[ ]:


cm = confusion_matrix(y_test, y_pred)
print(cm)

# In[ ]:


import matplotlib.pyplot as plt
# Create a heatmap from the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])

plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix Heatmap')
plt.show()

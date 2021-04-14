#!/usr/bin/env python
# coding: utf-8

# In[188]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.externals.six import StringIO 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from sklearn.ensemble import RandomForestClassifier


# In[7]:


filename = r"C:\Users\abhib\Desktop\diabetes.csv"
data=pd.read_csv(filename)
#dropping 0 values
data = data.drop(data[data['Glucose'] == 0].index)
data = data.drop(data[data['SkinThickness'] == 0].index) 
data = data.drop(data[data['BloodPressure'] == 0].index) 
data = data.drop(data[data['BMI'] == 0].index)
data = data.drop(data[data['Insulin'] == 0].index)


# In[8]:


data['Outcome'].describe()


# In[9]:


data['Outcome'].hist(figsize=(7,7))


# In[10]:


data['Pregnancies'].describe()


# In[12]:


data['Pregnancies'].hist(figsize=(7,7))


# In[14]:


data['Glucose'].describe()


# In[15]:


data['Glucose'].hist(figsize=(7,7))


# In[16]:


data['BloodPressure'].describe()


# In[17]:


data['BloodPressure'].hist(figsize=(7,7))


# In[18]:


data['SkinThickness'].describe()


# In[20]:


data['SkinThickness'].hist(figsize=(7,7))


# In[21]:


data['Insulin'].describe()


# In[22]:


data['Insulin'].hist(figsize=(7,7))


# In[23]:


data['BMI'].describe()


# In[24]:


data['BMI'].hist(figsize=(7,7))


# In[25]:


data['DiabetesPedigreeFunction'].describe()


# In[26]:


data['DiabetesPedigreeFunction'].hist(figsize=(7,7))


# In[27]:


data['Age'].describe()


# In[28]:


data['Age'].hist(figsize=(7,7))


# In[29]:


def plot_diabetic_per_attribute(data, feature):
    grouped_by_Outcome = data[feature].groupby(data["Outcome"])
    diabetic_per_feature = pd.DataFrame({"Diabetic": grouped_by_Outcome.get_group(1),
                                        "Not Diabetic": grouped_by_Outcome.get_group(0),
                                        })
    hist = diabetic_per_feature.plot.hist(bins=60, alpha=0.6)
    hist.set_xlabel(feature)
    plt.show()


# In[30]:


plot_diabetic_per_attribute(data, "Age")


# In[31]:


plot_diabetic_per_attribute(data, "Glucose")


# In[32]:


plot_diabetic_per_attribute(data, "BMI")


# In[33]:


import seaborn as sns 

corrmat = data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, cbar=True, annot=True, square=True, vmax=.8);


# In[34]:


cols = ['Pregnancies','Glucose','BloodPressure','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
sns.pairplot(data[cols], size = 2.5)
plt.show();


# In[146]:


X = data.iloc[:,:7]
y = data.iloc[:,8]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
from sklearn.decomposition import PCA
pca = PCA(3)
pca.fit(X)
X = pca.transform(X)
print(pca.n_components_)


# In[147]:


pca.components_


# In[167]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf = DecisionTreeClassifier(criterion="entropy", max_depth=4)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[187]:


cv = KFold(n_splits=10, random_state=1, shuffle=True)

model=MLPClassifier(max_iter=30000).fit(X_train,y_train)
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))


# In[191]:


model=RandomForestClassifier().fit(X_train,y_train)
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))


# In[ ]:





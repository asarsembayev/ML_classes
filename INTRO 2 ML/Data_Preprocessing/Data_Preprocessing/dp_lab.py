
# coding: utf-8

# # Import the libraries

# In[87]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# # Import the dataset

# In[88]:


dataset = pd.read_csv('Data.csv')


# In[89]:


print(dataset)


# # Make a matrix of features by splitting the dependent and independent variables
# ## The dependent variable in this case is the last column

# In[90]:


# This is the matrix of features (independent variables)
X = dataset.iloc[:,:-1].values


# In[91]:


# This is the dependent variables column
y = dataset.iloc[:,-1].values
# notice that you can either index as 'iloc[:,-1]' , or as iloc[:,3]. Both will work equally


# In[92]:


print(dataset)


# ### from the table above you may notice that there are two cells with missing data. One at 'Salary' and the other at 'Age'
# #### You may handle this by two possible solutions:
# #### 1) by deleting the lines w/ missing data. However, this is a dangerous way since these lines may contain crucial data
# #### 2) by putting the mean of the columns values
# ### We will try the second option

# ### For this purpose we import the Imputer class from the sklearn library

# In[93]:


# Taking care of missing data
from sklearn.preprocessing import Imputer


# ### and we also need to make an instance of the class
# ### In order to make this object we have to take care of the parameters for the Imputer() class. In Jupyter you can press Shift+Tab when pointing at brackets () and take a look at the pop-up window showing you the possible parameters to input
# ### you can also find the classes documentation by following this link: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html

# In[94]:


imputer = Imputer(missing_values='NaN', strategy= 'mean', axis = 0)


# ### Now it is a time to apply our created object on our data by using fit() method. Notice that we need to apply it not on the whole X dataset, but only a portion of it

# In[95]:


imputer = imputer.fit(X[:, 1:3])


# ### In order to apply it on X, use transform method

# In[96]:


X[:, 1:3] = imputer.transform(X[:,1:3])


# In[97]:


X


# ### now please check if the library was right in computing the mean (either in Excel or manually)
# 
# ## We are done with missing data. Now let's move on towards the categorical data

# In[98]:


# Encoding the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[99]:


# make an object of the class
labelEncoderX = LabelEncoder()


# In[100]:


X[:,0] = labelEncoderX.fit_transform(X[:,0])


# In[101]:


X


# ### Now you can see that we got rid of the String values in 'Countries' and now it makes more sense in mathematical terms. However, there might appear some problems with these values? Can you guess which kind of? Take a look at the values...

# ### This problems can be tackled by encoding. There are several types of encoding: dummy, one-hot encoder etc.

# In[102]:


onehotEncoder = OneHotEncoder(categorical_features = [0])


# In[103]:


X


# In[104]:


X = onehotEncoder.fit_transform(X).toarray()


# In[117]:


print(X)


# ### Now let's encode 'y' with label encoder. The reason that we use label encoder here is because we know that the output is binary

# In[118]:


labelEncoderY = LabelEncoder()
y = labelEncoderY.fit_transform(y)


# In[119]:


y


# # Now we should split the dataset

# In[121]:


from sklearn.cross_validation import train_test_split


# In[122]:


# random state is a pseudo-random number generator
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[131]:


print(len(X_train))


# In[132]:


print(len(X_test))


# # Now let's do the Feature Scaling on numerical values like 'Salary' and 'Age'

# In[135]:


from sklearn.preprocessing import StandardScaler


# In[138]:


sc_X = StandardScaler()


# In[139]:


X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[141]:


print(X_train)


# In[142]:


print(X_test)


# ## We don't need to apply feature scaling for 'y' because it is a categorical variable of binary type, but for the regression problems there is a need to apply the scaling

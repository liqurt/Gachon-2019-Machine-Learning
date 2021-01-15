
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV
import warnings

warnings.filterwarnings('ignore')


# In[19]:


df = pd.read_csv('bank-full.csv', sep=';') # 45211
print(df.columns)


# In[20]:


# Data Preprocessing
# 1) Data Restructuring - Table Vertical Decomposition
# Delete unusable columns
columns = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan','balance']
df = df[columns]
df = df.astype({"age":int,"balance": float})
df = df[df.marital != 'divorced']

print(df.columns)
print(len(df))


# In[22]:


import matplotlib.pyplot as plt
data = df["balance"].values
plt.boxplot(np.array(data).astype(np.float))
plt.show()


# In[12]:


# 2) Data Restructuring - Data Value Changes
# Cleaning dirty data 

# Missing Data -> Change "unknown" to NaN
df = df.replace('unknown',np.nan)
print("* Missing data before removal *")
print("Total data : ",len(df))
df.isnull().sum()
df = df.dropna() # Delete Missing Data
print("* Missing data after removal *")
print("Total data : ",len(df))
df.isnull().sum() # Check Missing Data

# Outliers
df = df[df["balance"]<=40000]
df = df[df["balance"]>=-500]

# Delete Redundancy
print("* data before removing redundancy *",len(df))
df = df.drop_duplicates()
print("* data after removing redundancy *",len(df))
print(df.head(10))


# In[13]:


# 3) Feature Engineering - Feature Creation
# Label Encoding
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
labelEncoder.fit(df['job'])
df['job'] = labelEncoder.transform(df['job'])

labelEncoder.fit(df['marital'])
df['marital'] = labelEncoder.transform(df['marital'])

labelEncoder.fit(df['education'])
df['education'] = labelEncoder.transform(df['education'])

labelEncoder.fit(df['default'])
df['default'] = labelEncoder.transform(df['default'])

labelEncoder.fit(df['housing'])
df['housing'] = labelEncoder.transform(df['housing'])

labelEncoder.fit(df['loan'])
df['loan'] = labelEncoder.transform(df['loan'])

feature = df.drop(columns=['marital'])
label = df["marital"]
print(label)


# In[14]:


# Data value changes - Normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
feature = scaler.fit_transform(feature)
print(feature)


# In[15]:


# split dataset into train and test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(feature,label, test_size=0.3, random_state=0)
print(x_train)


# In[16]:


# 3) Feature Engineering - Feature Reduction
# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=4) # 7 columns -> 4 columns
x_train = pca.fit_transform(x_train)
print(x_train)


# In[17]:


# Decision Tree

from sklearn.tree import DecisionTreeClassifier

decision_param = {'criterion' : ['gini', 'entropy'],
                  'min_samples_split' : [2, 10, 20],
                 'max_depth' : [1, 5, 10, 15],
                  'min_samples_leaf' : [1, 5, 10],
                  'max_leaf_nodes' : [5, 10, 20, 30, 40, 50],
                 'random_state' : [30, 50, 100]}

# Build three classifier
decision = DecisionTreeClassifier()

# Bulid GridSearchCV which find the parameter combination with the highest score
grid_decision = GridSearchCV(decision, param_grid = decision_param, scoring = 'accuracy', cv = 5)

# Training Decision Tree GridSearchCV
grid_decision.fit(x_train, y_train)

# The highest score and the best parameter combination
print("* Decision Tree *")
print("Best Parameter :", grid_decision.best_estimator_) # return the best combination parameter
print("High Score :", grid_decision.best_score_) # return the best parameter with the highest score

# Test GridSearchCV using test data set
y_pred_decision = grid_decision.best_estimator_.predict(x_test)


# In[ ]:


# Logistic Regression

from sklearn.linear_model import LogisticRegression

# Logistic Regression Parameter
logistic_param = {'solver' : ['liblinear', 'lbfgs', 'sag'],
                  'max_iter' : [50, 100, 200]}

# Build logisticRegression classifier
logistic = LogisticRegression()

# Bulid GridSearchCV which find the parameter combination with the highest score
grid_logistic = GridSearchCV(logistic, param_grid = logistic_param, scoring = 'accuracy', cv = 10)

# Training Decision Tree GridSearchCV
grid_logistic.fit(x_train, y_train)

# The highest score and the best parameter combination
print("\n* Logistic Regression *")
print("Best Parameter :", grid_logistic.best_estimator_) # return the best combination parameter
print("High Score :", grid_logistic.best_score_) # return the best parameter with the highest score

# Test GridSearchCV using test data set
y_pred_logistic = grid_logistic.best_estimator_.predict(x_test)


# In[178]:


# SVM

from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV

svm_param = {'kernel' : ['linear','rbf','sigmoid'],
                 'gamma' : [10, 100],
                 'C' : [0.1, 1.0, 10.0]}

# Build svm classifier
svm = SVC()

# Bulid GridSearchCV which find the parameter combination with the highest score
grid_svm = GridSearchCV(svm, param_grid = svm_param, scoring = 'accuracy', cv = 10)

# Training Decision Tree GridSearchCV
grid_svm.fit(x_train, y_train)

# The highest score and the best parameter combination
print("\n* SVM *")
print("Best Parameter :", grid_svm.best_estimator_) # return the best combination parameter
print("High Score :", grid_svm.best_score_) # return the best parameter with the highest score

# Test GridSearchCV using test data set
y_pred_svm = grid_svm.best_estimator_.predict(x_test)


# In[180]:





# In[23]:


# Score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

print("** Decision Tree **")
print(classification_report(y_test, y_pred_decision, target_names=['class 0', 'class 1', 'class 2']))

print("** Logistic Regression **")
print(classification_report(y_test, y_pred_logistic, target_names=['class 0', 'class 1', 'class 2']))

print("** SVM **")
print(classification_report(y_test, y_pred_svm, target_names=['class 0', 'class 1', 'class 2']))

score = []
score.append(grid_decision.best_score_)
score.append(grid_logistic.best_score_)
score.append(grid_svm.best_score_)

classifier = ['Decision Tree', 'Logistic Regression', 'SVM']
plt.bar(classifier, score)
plt.show()


# In[24]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred_decision))
print(confusion_matrix(y_test, y_pred_logistic))
print(confusion_matrix(y_test, y_pred_svm))

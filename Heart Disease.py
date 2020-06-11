#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer 
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import scipy.stats as stats
from scipy.stats import chi2_contingency


# In[2]:


class ChiSquare:
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None #P-Value
        self.chi2 = None #Chi Test Statistic
        self.dof = None
        
        self.dfObserved = None
        self.dfExpected = None
        
    def _print_chisquare_result(self, colX, alpha):
        result = ""
        if self.p<alpha:
            result="{0} is IMPORTANT for Prediction".format(colX)
        else:
            result="{0} is NOT an important predictor. (Discard {0} from model)".format(colX)

        print(result)
        
    def TestIndependence(self,colX,colY, alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)
        
        self.dfObserved = pd.crosstab(Y,X) 
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof 
        
        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)
        
        self._print_chisquare_result(colX,alpha)


# In[54]:


dataset = pd.read_csv('cleveland.csv', sep = ',', names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'])


# In[55]:


dataset.head(10)


# In[5]:


processed_dataset = dataset.replace({'num': [2,3,4]},1)


# In[6]:


processed_dataset.head(10)


# In[7]:


import seaborn as sns

sns.countplot(x="num",data=processed_dataset,palette="bwr")
plt.show()


# In[8]:


sns.countplot(x='sex', data=processed_dataset, palette="mako_r")


# In[53]:


# processed_dataset.isnull().sum()
processed_dataset.num.value_counts()


# In[10]:


#Dataset dengan Mengganti Mean
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
idf=pd.DataFrame(imputer.fit_transform(processed_dataset))
idf.columns=processed_dataset.columns
idf.index=processed_dataset.index

dataset_mean_impute_X = idf.drop('num',axis=1)
dataset_mean_impute_y = idf['num']

#Menghapus null values NA
processed_dataset.dropna(subset = ["ca"], inplace=True)
processed_dataset.dropna(subset = ["thal"], inplace=True)
dataset_dropped_nullX = processed_dataset.drop('num',axis=1)
dataset_dropped_nullY = processed_dataset['num']


# In[35]:


#Initialize ChiSquare Class Mean dataset
cTMean = ChiSquare(idf)

#Feature Selection
testColumns = [ 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',]
for var in testColumns:
    cTMean.TestIndependence(colX=var,colY="num" )

x_chi_mean = dataset_mean_impute_X.drop(['age','trestbps','chol','fbs'],axis=1)
y_chi_mean = dataset_mean_impute_y
x_chi_drop = dataset_dropped_nullX.drop(['age','trestbps','chol','fbs'],axis=1)
y_chi_drop = dataset_dropped_nullY


# In[56]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
#Model 0:
# - UnProcessed data
X = processed_dataset.drop('num',axis=1)
Y = processed_dataset['num']

print(Y.head(10))
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size= 0.25, random_state=0)
X_train = sc_x.fit_transform(X_train)
X_test  = sc_x.transform(X_test)

# Model 1 :
# - Changed Missing Values with mean
# - find features with Chi Squared 
# - Standard Scaler
X_mean_Train, X_mean_Test, y_mean_Train, y_meanTest = train_test_split(x_chi_mean, y_chi_mean, test_size=0.25, random_state=0)
X_mean_Train = sc_x.fit_transform(X_mean_Train)
X_mean_Test = sc_x.transform(X_mean_Test)


# Model 2: 
# - Dropped Missing values
# - find features with Chi Squared 
# - Standard Scaler
Xdrop_train_chi, Xdrop_test_chi,ydrop_train_chi,ydrop_test_chi = train_test_split(x_chi_drop,y_chi_drop,test_size=0.25, random_state =0)

Xdrop_train_chi = sc_x.fit_transform(Xdrop_train_chi)
Xdrop_test_chi = sc_x.transform(Xdrop_test_chi)


# In[90]:


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=7, p=2, metric='minkowski')
clf.fit(X_train,y_train)
yPred = clf.predict(X_test)
mse = mean_squared_error(y_test,yPred)
accuracy = accuracy_score(y_test,yPred)
print("Accuracy = ", accuracy)
num_neighbors = 10


# In[98]:



clf_chi_mean = KNeighborsClassifier(n_neighbors=7, p=2, metric='minkowski')
clf.fit(X_mean_Train,y_mean_Train)
yPred_chi = clf.predict(X_mean_Test)
accuracy_chi = accuracy_score(y_meanTest,yPred_chi)
print(accuracy_chi)


# In[94]:


clf_chi_drop = KNeighborsClassifier(n_neighbors=num_neighbors, p=2, metric='minkowski')
clf.fit(Xdrop_train_chi,ydrop_train_chi)
yPred__dropchi = clf.predict(Xdrop_test_chi)
accuracy_chi = accuracy_score(ydrop_test_chi,yPred__dropchi)
print(accuracy_chi)


# In[ ]:





# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # ETEL Coding Challenge

# ## Data Loading

# In[3]:


# This code assumes the following packages are installed:
# - pandas, including openpyxl
# - numpy
# - matplotlib
# - sklearn

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier


# In[4]:


df = pd.read_excel('./bank_loan_dataset.xlsx', header=2, usecols='B:W')


# In[5]:


# Parsing the attribute keys so they can later be replaced in the dataframe
att = pd.read_excel('./bank_loan_dataset.xlsx', sheet_name='Data Dictionary', usecols='B').to_string(index=False)
att_key ={}
regex = re.compile('A.{2,3}:')
for line in att.splitlines():
    a = re.split(r'A.{2,8}:', line)
    # removing lines that have no keys
    if len(regex.findall(line)) > 0 and len(a) > 1: 
        # removing duplicates (from lines where there's 'if not')
        A_num = list(dict.fromkeys(regex.findall(line))) 
        # removing instances with 'if not'
        description = [i for i in a[1:] if i !=' if not '] 
        for i, j in zip(A_num, description):
            att_key[i.strip(':')] = j.replace('\xa0','').strip(' ')
#print(att_key)


# In[6]:


# Parsing the attribute legend. 
# This is not used later in this code but could be useful otherwise.
att_legend=[]
for line in att.splitlines():
    m = re.match('\ *Attribute', line)
    if m:
        a = re.split(r'A.{2,3}:', line.strip())
        att_legend.append(a[0].split(':'))
att_legend = dict(att_legend)

for i in att_legend.keys():
    att_legend[i] = att_legend[i].strip()
#print(att_legend)


# In[7]:


# Replacing the attribute keys with their values
df = df.replace(to_replace=att_key.keys(), value=att_key.values()).set_index('Customer_ID')


# ## Analysis

# In[8]:


print('Printing a summary to check data types and look for missing values:')
print(df.info())


# In[21]:


def plot_all_distr(dataframe):
    """Plotting all variable distributions, using bar charts for categorical variables
    and histograms for numerical variables"""
    num_plots = dataframe.shape[1]
    defaulted = dataframe[dataframe['Default_On_Payment']==1].shape[0]
    not_defaulted = dataframe[dataframe['Default_On_Payment']==0].shape[0]
    fig, axes = plt.subplots(num_plots,figsize=(18,num_plots*6), sharey=False)
    i=0
    for col in dataframe.columns:
        if df[col].dtype == 'O':
            labels, counts = np.unique(np.array(dataframe[dataframe['Default_On_Payment']==0][col]), return_counts=True)
            counts_prop = counts/counts.sum()
            axes[i].bar(labels, counts_prop, width=-0.4, align='edge', color='b', label='No Default on Loan')
            labels, counts = np.unique(np.array(dataframe[dataframe['Default_On_Payment']==1][col]), return_counts=True)
            counts_prop = counts/counts.sum()
            axes[i].bar(labels, counts_prop, width=0.4, align='edge', color='r', label='Defaulted on Loan')
            axes[i].set_xticks(labels)
            axes[i].set_xticklabels(labels, rotation = -3)
            axes[i].set_ylabel('Proportion of Customers')
        elif df[col].dtype == 'int64':
            c, bins, p = axes[i].hist(dataframe[dataframe['Default_On_Payment']==0][col], color='b', alpha=0.8, label='No Default on Loan')
            axes[i].hist(dataframe[dataframe['Default_On_Payment']==1][col], color='r', bins=bins, alpha=0.8, label='Defaulted on Loan')
            axes[i].set_xticks(bins)
            axes[i].set_ylabel('Number of Customers')
        axes[i].set_title(col)
        axes[i].legend()
        i +=1
    plt.show()

print('Plotting the distribution of all variables:')
plot_all_distr(df)


# From the plots above we can derive a few useful observations:
# - Customers who defaulted have generally a lower amount in their checking account. However, roughly half the customers that did not default had no checking account.
# - The loans that did not default have mostly shorter durations. The duration of loans that defaulted tends more towards a uniform distribution, therefore the proportion of loans that defaulted is higher for higher durations.
# - The credit amount distribution shows the proportion of defaulted to non-defaulted loans becomes higher with higher amounts.
# - Buying a new car is the most common purpose of loans that defaulted. This could be due to the higher cost of the car, requiring higher credit amounts and longer loan durations.
# - There is little margin between a default and no default when considering the other variables.

# In[22]:


# To further analyse the data all categorical variables are transformed
# to sets of binary variables using one-hot encoding
df_onehot = pd.get_dummies(data=df, columns=[col for col in df.columns if df[col].dtype=='O'], drop_first=False)


# In[23]:


print('The cross-correlation between the variables is calculated and plotted:')
# This can show if there is a relationship between any of the variables and
# with Default_on_Payment

fig, ax = plt.subplots(1,1, figsize=(12,12))
corr_mat = ax.matshow(df_onehot.corr())
plt.xticks(range(len(df_onehot.corr().columns)), df_onehot.corr().columns, rotation=90)
plt.yticks(range(len(df_onehot.corr().columns)), df_onehot.corr().columns)
plt.colorbar(corr_mat)
plt.title('Cross-correlations between all variables')
plt.show()


# The plot above looks quite busy and nothing obvious is standing out. To extract more meaningful information the we'll look for the pairs of variables with a high correlation between them.

# In[24]:


def high_corr_pairs(df, threshold):
    """This selects the pairs of variables that have a cross-correlation above
    a threshold"""
    results =[]
    df_corr = df_onehot.corr()
    cols = df_corr.columns
    for col in cols:
        for row in cols:
            corr_val = abs(df_corr.loc[row,col])
            if (corr_val > threshold) and (corr_val < 1):
                results.append(tuple(sorted((row, col, str(corr_val)))))
    # Removing duplicate pairs and sorting            
    results = sorted(list(set(results)),reverse=True) 
    results = filter_onehot_results(df, results)
    return filter_onehot_results(df, results)

def filter_onehot_results(df, results):
    """This function filters out results that have a high correlation because of
    the one-hot encoding"""
    for col in df.columns:
        if df[col].dtype == 'O':
            for c, i, j in results:
                if col in i and col in j:
                    results.remove((c,i,j))
    return results

print('Correlation coefficents and their respective pairs of variables:')
for result in high_corr_pairs(df, 0.5):
    print(result)


# The above result doesn't show anything unexpected:
# - there's a correlation between the amount and the duration of the loan
# - there's a correlation between the credit history and the amount of credits held

# ## Modelling

# In[27]:


# The one-hot encoding is re-run as we need to drop a binary variable for each categorical one
df_onehot = pd.get_dummies(data=df, columns=[col for col in df.columns if df[col].dtype=='O'], drop_first=True)


# In[28]:


# Preparing data and splitting
X = df_onehot.drop(columns='Default_On_Payment').values
y = df_onehot.loc[:,'Default_On_Payment'].values

# 10% of the data is kept for the final test.
# The rest will be used for training and validation
X_train_v, X_test, y_train_v, y_test = train_test_split(X, y, test_size=0.1, random_state=0)


# In[29]:


# A class is defined to construct and evaluate the classifier
# Principle component analysis (PCA) is used to reduce the amount of
# features going in to the model
# The train/validation data is used for K-fold crossvalidation.
# In this case, for k=5, it means 80% of the data will be used for training
# while 20% is reserved for validation of the model.

class Loan_classifier:
    def __init__(self, X_train_v, y_train_v):
        self.X_train_v = X_train_v
        self.y_train_v = y_train_v
    
    def prep_setup(self, X_train, pca_comp):
        """Setting up the scaler and PCA, needs to be run first but is included
        in the train function"""
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler = self.scaler.fit(X_train)
        X_train_sc = self.scaler.transform(X_train)
        self.pca = PCA(n_components=pca_comp)
        self.pca = self.pca.fit(X_train_sc)
    
    def preproc(self, X):
        X_sc = self.scaler.transform(X)
        X_proj = self.pca.transform(X_sc)
        return X_proj

    def cross_validation(self, model, k=5, pca_comp=2):
        kf = KFold(n_splits=k)
        train_acc_list = []
        valid_acc_list = []
        for train_index , valid_index in kf.split(self.X_train_v):
            # Split the data in training and validation
            X_train , X_valid = self.X_train_v[train_index], self.X_train_v[valid_index]
            y_train , y_valid = self.y_train_v[train_index], self.y_train_v[valid_index]
            # Fitting the model, includes preprocessing
            train_acc = self.train(X_train, y_train, model, pca_comp=2) 
            # Predictions for the validation data, includes preprocessing
            y_pred = self.predict(X_valid) 
            valid_acc = np.mean(y_pred==y_valid)
            train_acc_list.append(train_acc) 
            valid_acc_list.append(valid_acc)
        return sum(train_acc_list)/k, sum(valid_acc_list)/k
    
    def train(self, X_train, y_train, model, pca_comp=2):
        self.prep_setup(X_train, pca_comp)
        X_train_proj = self.preproc(X_train)
        self.model = model.fit(X_train_proj, y_train)
        return self.model.score(X_train_proj, y_train)
        
    def predict(self, X):
        X_proj = self.preproc(X)
        y_pred = self.model.predict(X_proj)
        return y_pred


# In[30]:


# 4 different models will be tested:
# - Support Vector Machine
# - K nearest neighbour
# - Gradient boosting
# - Random Forest
# These are chosed as thery're commonly used for classification.
# Their hyperparamters have been tuned beforehand to produce good results.

model_svm  = svm.SVC(C=100, gamma=300)
model_knn  = KNeighborsClassifier(n_neighbors=1)
model_gb = GradientBoostingClassifier(n_estimators=150, learning_rate=1, min_samples_leaf=2)
model_rfc = RandomForestClassifier(n_estimators=20, min_samples_leaf=1)

models = [model_svm, model_knn, model_gb, model_rfc]

# Note: there's a high risk of overfitting while using only 1 neighbour with knn,
# however validation results look good so it's deemed acceptable in this case


# In[31]:


Loans_01 = Loan_classifier(X_train_v, y_train_v)

print('Several combinations of models and number of features (out of PCA) are tested:')

for mod in models:
    print('\n',mod)
    for p in [1, 2, 4, 10]:
        print('Number of features:',p)
        print('Training and validation accuracy:')
        print(Loans_01.cross_validation(mod, pca_comp=p))


# In[35]:


# The model (and number of features) with the highest validation accuracy
# is selected as final model. The test data is now used for the final evaluation.

print('We can now test our final model:')
Loans_01.train(X_train_v, y_train_v, model_gb, 1)

y_pred = Loans_01.predict(X_test)

print('Final test accuracy:', np.mean(y_pred==y_test))
print('Test predictions:\n', y_pred, '\n')

print('Confusion matrix:')
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))


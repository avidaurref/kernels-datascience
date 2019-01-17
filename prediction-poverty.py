# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:51:28 2018

@author: Alvaro
"""

# Load libraries
import os
import numpy as np
import pandas as pd

#import matplotlib as plt
from sklearn.ensemble import (RandomForestClassifier,GradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.metrics import log_loss

# Load Own functions
def standardize( df, numeric_online=True):
    numeric = df.select_dtypes(include=['int64','float64'])
    
    #substract mean and divide by std
    df[numeric.columns] = ( numeric - numeric.mean() ) / numeric.std() 
    
    return df

def pre_process_data(df, enforce_cols=None):
    print("Input shape:\t{}".format(df.shape))
    
    df = standardize(df)
    print("After standarization {}".format(df.shape))
    
    #create dummy variables for categoricals
    df = pd.get_dummies(df)
    print("After converting categoricals:\t{}".format(df.shape))
    
    # match test set and training set columns
    if enforce_cols is not None:
        to_drop = np.setdiff1d(df.columns, enforce_cols)
        to_add = np.setdiff1d(enforce_cols, df.columns)
        
        df.drop(to_drop, axis=1, inplace=True)
        df = df.assign(**{c: 0 for c in to_add})
        
    df.fillna(0, inplace=True)
    
    return df

def train_model(features, labels, **kwargs):
    
    # instantiate model
    #model = RandomForestClassifier(n_estimators=50, random_state=0)
    #model = SVC(C = 10, probability = True)
    model = GradientBoostingClassifier(n_estimators= 50, random_state = 0)
    
    # train model
    model.fit(features, labels)
    
    #get a (not-very-useful) sense of performance
    accuracy = model.score(features, labels)
    print(f"In-sample accuracy: {accuracy:0.2%}")
    
    return  model

def make_country_sub(preds, test_feat, country):
    
    # get just the poor probabilities
    country_sub = pd.DataFrame(data=preds[:, 1],  # proba p=1
                               columns=['poor'], 
                               index=test_feat.index)

    
    # add the country code for joining later
    country_sub["country"] = country
    
    return country_sub[["country", "poor"]]


# Load files
DATA_DIR = 'C:\\Users\\Alvaro\\Acciones\\DSC002\\data\\'
data_paths = {'A': {'train': os.path.join(DATA_DIR, 'A_hhold_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'A_hhold_test.csv')}, 
              'B': {'train': os.path.join(DATA_DIR, 'B_hhold_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'B_hhold_test.csv')}, 
              'C': {'train': os.path.join(DATA_DIR, 'C_hhold_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'C_hhold_test.csv')}}
              
# load training data
a_full = pd.read_csv(data_paths['A']['train'], index_col='id')
b_full = pd.read_csv(data_paths['B']['train'], index_col='id')
c_full = pd.read_csv(data_paths['C']['train'], index_col='id')
# load test data
a_test = pd.read_csv(data_paths['A']['test'], index_col='id')
b_test = pd.read_csv(data_paths['B']['test'], index_col='id')
c_test = pd.read_csv(data_paths['C']['test'], index_col='id')

# Explore shape of data
a_full.shape
b_full.shape
c_full.shape

b_full.poor.value_counts().plot.bar(title='Number of Poor for country A')
b_full.poor.value_counts().plot.bar(title='Number of Poor for country A')

# Process data
aT_full = pre_process_data(a_full.drop('poor', axis=1))
aT_full = aT_full.assign(poor = a_full.poor) 
bT_full = pre_process_data(b_full.drop('poor', axis=1))
bT_full = bT_full.assign(poor = b_full.poor) 
cT_full = pre_process_data(c_full.drop('poor', axis=1))
cT_full = cT_full.assign(poor = c_full.poor) 

# Split train and validation data
aT_trn = aT_full.sample(frac = 0.5, random_state = 123)
aT_val = aT_full.drop(aT_trn.index)
bT_trn = bT_full.sample(frac = 0.5, random_state = 123)
bT_val = bT_full.drop(bT_trn.index)
cT_trn = cT_full.sample(frac = 0.5, random_state = 123)
cT_val = cT_full.drop(cT_trn.index)

# Training
model_a = train_model(aT_trn.drop('poor', axis=1), np.ravel(aT_trn.poor) )
model_b = train_model(bT_trn.drop('poor', axis=1), np.ravel(bT_trn.poor) )
model_c = train_model(cT_trn.drop('poor', axis=1), np.ravel(cT_trn.poor) )

# Training full
model_a = train_model(aT_full.drop('poor', axis=1), np.ravel(aT_full.poor) )
model_b = train_model(bT_full.drop('poor', axis=1), np.ravel(bT_full.poor) )
model_c = train_model(cT_full.drop('poor', axis=1), np.ravel(cT_full.poor) )


# Prediction
a_preds = pd.DataFrame(model_a.predict_proba(aT_val.drop('poor', axis=1)))
b_preds = pd.DataFrame(model_b.predict_proba(bT_val.drop('poor', axis=1)))
c_preds = pd.DataFrame(model_c.predict_proba(cT_val.drop('poor', axis=1)))

# Log loss
a_logloss = log_loss(aT_val.poor, a_preds)
b_logloss = log_loss(bT_val.poor, b_preds)
c_logloss = log_loss(cT_val.poor, c_preds)

mean_logloss = np.mean([a_logloss,b_logloss,c_logloss])
print(mean_logloss)

# Submission------------------------------------------

# Read test data
a_test = pd.read_csv(data_paths['A']['test'], index_col='id')
b_test = pd.read_csv(data_paths['B']['test'], index_col='id')
c_test = pd.read_csv(data_paths['C']['test'], index_col='id')

# process the test data
a_test = pre_process_data(a_test, enforce_cols=aT_trn.drop('poor', axis=1).columns)
b_test = pre_process_data(b_test, enforce_cols=bT_trn.drop('poor', axis=1).columns)
c_test = pre_process_data(c_test, enforce_cols=cT_trn.drop('poor', axis=1).columns)

# predict
a_preds = model_a.predict_proba(a_test)
b_preds = model_b.predict_proba(b_test)
c_preds = model_c.predict_proba(c_test)

# convert preds to data frame
a_sub = make_country_sub(a_preds, a_test, 'A')
b_sub = make_country_sub(b_preds, b_test, 'B')
c_sub = make_country_sub(c_preds, c_test, 'C')

submission = pd.concat([a_sub, b_sub, c_sub])

submission.to_csv(os.path.join(DATA_DIR,'submission4.csv'))
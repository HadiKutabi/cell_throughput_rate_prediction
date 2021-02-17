#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from  datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import geopy

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor 
from sklearn.linear_model import Lasso 
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


from multiprocessing import cpu_count
from sklearn.model_selection import learning_curve

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('darkgrid')
sns.set_context('talk')

geolocator = geopy.Nominatim(user_agent='TEST')

def unix_to_datetime(instance): 
    return pd.Timestamp(datetime.fromtimestamp(instance).strftime('%Y-%m-%d %H:%M:%S'))

def unix_to_date(instance):
    return pd.Timestamp(datetime.fromtimestamp(instance).strftime('%Y-%m-%d'))



def featurize_datetime(df):
    
    df['hour'] = df.rawTimesamp.dt.hour
    df['week'] = df.rawTimesamp.dt.week
    df['year'] = df.rawTimesamp.dt.year
    df['dayofweek'] = df.rawTimesamp.dt.dayofweek
    df['dayofmonth'] = df.rawTimesamp.dt.day
    df['month'] = df.rawTimesamp.dt.month

    return df


def encode_day_time_cat(df, check_col,col_name):
    df = featurize_datetime(df)
    df[col_name] = int()
    
    df.loc[(df[check_col] >= 5) & (df[check_col] < 10) , col_name] = 1 
    df.loc[(df[check_col] >= 10) & (df[check_col] < 12) , col_name] = 2 
    df.loc[(df[check_col] >= 12) & (df[check_col] < 15) , col_name] = 3 
    df.loc[(df[check_col] >= 15) & (df[check_col] < 18) , col_name] = 4 
    
    return df

def onehot_enc(df, col):
    enc_oh = OneHotEncoder(sparse=False)
    enc_oh.fit(df[[col]])
    category_columns = np.concatenate(enc_oh.categories_)
    encoded_features = enc_oh.transform(df[[col]])

    df[category_columns] = pd.DataFrame(
            encoded_features,
            columns=category_columns,
            index=df.index)
    return df.drop([col], axis =1)

def means_and_std(timeseries, rolling_col= '', target_col = ''):
    throughput_mean = []
    throughput_std = []
    throughput_var = []

    for ci in timeseries[rolling_col].unique():
        
        ts_slice = timeseries.loc[timeseries[rolling_col] == ci]
        
        roll_window = ts_slice.shape[0]
        
        means = ts_slice[target_col].rolling(roll_window,min_periods = 1).mean()
        stds = ts_slice[target_col].rolling(roll_window,min_periods = 1).std()
        var = ts_slice[target_col].rolling(roll_window,min_periods = 1).var()
        
        for mean in means:
            throughput_mean.append(mean)
        for std in stds:
            throughput_std.append(std)
        for vr in var:
            throughput_var.append(vr)
           
            
    mean_name = target_col + '_mean'
    std_name = target_col + '_std'
    varame = target_col + '_var'
         
    
    timeseries[mean_name] = throughput_mean
    timeseries[std_name] = throughput_std
    timeseries[varame]= throughput_var
    
  #timeseries[std_name] = timeseries[std_name].fillna(0)
  # timeseries[std_name] = timeseries[std_name].fillna(0) # this is why the first rows will be  0
  # timeseries[varame] = timeseries[varame].fillna(0)
    
    return timeseries.fillna(0)

def shift_group_features_(timeseries, group_by ='', shift_col1 ="",
                          shift_col2 ="", shift_col3=""):
                          
    
    means_shifted = []
    stds_shifted = [] 
    vars_shifted = []
    
    groups = timeseries[group_by].unique() # the shift will be done for each ci solely 
    
    for group in groups:
        entitiy = timeseries.loc[timeseries[group_by] == group]
        new_col1 = entitiy[shift_col1].shift(periods = 1, fill_value = 0)
        new_col2 = entitiy[shift_col2].shift(periods = 1, fill_value = 0)
        new_col3 = entitiy[shift_col3].shift(periods = 1, fill_value = 0)
        
        for mean, std, var in zip(new_col1, new_col2, new_col3):
            means_shifted.append(mean)
            stds_shifted.append(std)
            vars_shifted.append(var)
            
    timeseries[shift_col1] = means_shifted
    timeseries[shift_col2] = stds_shifted
    timeseries[shift_col3] = vars_shifted  

    return timeseries


def get_district(lat_lon):
    
    address = geolocator.reverse(lat_lon, zoom=14, timeout=None)
    dist = address[0]
    
    return dist.split(',')[0]


def regressor_rmse(model,x_train, x_test, y_train, y_test, model_name = ""):
    regressor = model
    regressor.fit(x_train,y_train)
    
    predictions = regressor.predict(x_test)
    
    score = np.sqrt(mean_squared_error(y_test, predictions))
    print("Model: {}".format(model_name), "\n",
         "RMSE: {}".format(round(score,2)))




def scale(features_df,trans_df , scaler):
    transformer = scaler.fit(features_df)
    return transformer.transform(trans_df)

def fit_xgb(x_tr, y_tr, n_estimators, max_depth, learning_rate, objective,
             booster, min_child_weight, reg_alpha,reg_lambda, 
           base_score, random_state):
    
    reg = xgb.XGBRegressor(n_estimators = n_estimators, max_depth  = max_depth,
                           learning_rate = learning_rate, 
                          objective =objective, booster = booster,
                           min_child_weight = min_child_weight,  reg_alpha= reg_alpha, 
                          reg_lambda = reg_lambda, base_score = base_score,
                          random_state = random_state)
    
    
    return reg.fit(x_tr, y_tr)

def fit_rf(x_tr, y_tr, n_estimators, max_depth,
             min_samples_split, max_samples, ccp_alpha, random_state):
    
    reg = RandomForestRegressor(n_estimators = n_estimators, max_depth  = max_depth,
                           min_samples_split = min_samples_split, max_samples = max_samples,
                          ccp_alpha = ccp_alpha, random_state = random_state)
    
    return reg.fit(x_tr, y_tr)

def rmse(regressor,x_test, y_test):
    preds = regressor.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print("RMSE: %f" % (rmse))
    return rmse

def plot_learning_curves(model, x, y, random_state,cv=3, scoring='',
                         scoring_unit='', model_name='',title='',
                         train_sizes=np.linspace(0.1, 1.0, 10),
                         n_jobs=1, ax=None):
    ds_sizes, train_scores, valid_scores = learning_curve(
            model, x, y, scoring=scoring, n_jobs=n_jobs,
            train_sizes=train_sizes, cv=cv, shuffle=False, random_state = random_state)
    
    
    
    # create dataframe for plot
    mean_train_scores = np.mean(train_scores, axis=1)
    mean_valid_scores = np.mean(valid_scores, axis=1)
    score = prettify_scoring_name(scoring)
    scores_df = pd.DataFrame({
        'Training Set Size': np.concatenate((ds_sizes, ds_sizes)),
        score: np.concatenate(
            (mean_train_scores, mean_valid_scores)),
        'Dataset': (['Training'] * train_sizes.shape[0] + 
                    ['Validation'] * train_sizes.shape[0])
    })

    # create plot
    axes = sns.lineplot(
        x='Training Set Size', y=score, hue='Dataset', data=scores_df, ax=ax)
    axes.set_title(model_name + '\n'+ title )
    
    

def prettify_scoring_name(scoring_name):

    words = scoring_name.split('_')
    return " ".join(list(map(lambda x: x.capitalize(), words)))


def visualize_prediction_value_ordered_examples(y_test, y_predicted, 
                                               title = '', file_path_name = '', 
                                               score = 0):

    df_prediction = pd.DataFrame({
            'Actual':y_test, 
            'Predicted':y_predicted
    })
     #reindexing 
    df_prediction = df_prediction.sort_values('Actual').reset_index(drop=True)
    
    # add the index as a column so we can later use melt
    df_prediction['Test Set Example'] = df_prediction.index

    # Melting
    df_lineplot = pd.melt(df_prediction, id_vars=['Test Set Example'],
                          value_vars=['Actual', 'Predicted'],
                          var_name='Curve', value_name='Target Variable')
    # plot comparison
    plt.figure(figsize=(15, 5))
    ax_lineplot = sns.lineplot(data=df_lineplot.sort_index(ascending=False, axis=0), 
                               y='Target Variable', x='Test Set Example', 
                               hue='Curve', style='Curve')
    ax_lineplot.set_title(title + '(RMSE ='  + str(round(score, 2)) + ')', loc='left' )
    plt.show()
    




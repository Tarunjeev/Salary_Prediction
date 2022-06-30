#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from math import gamma
from sklearn.metrics import mean_absolute_error
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from math import gamma


OUTPUT_TEMPLATE = (
    '"Output by running linear regression model" MAE:  {score_lm:.3g}\n'
    '"Output by running random forest" MAE:  {score_rf:.3g} \n'
    '"Output by running XGBoost" MAE:  {score_xg:.3g}\n'
)


def linear_regression(X_train,y_train,X_test,y_test):
    model_lm = LinearRegression()
    model_lm.fit(X_train, y_train)
    prediction_lm = model_lm.predict(X_test)
    result_lm = mean_absolute_error(y_test,prediction_lm)
    return result_lm


def random_forest(X_train,y_train,X_test,y_test):
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor()
    regressor.fit(X_train, y_train)
    np.mean(cross_val_score(regressor,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))
    model_rf = rf_tuning(X_train, y_train, regressor)
    prediction_rf = model_rf.predict(X_test)
    result_rf = mean_absolute_error(y_test,prediction_rf)
    return result_rf


def rf_tuning(X_train, y_train, regressor):
    params = {
    "n_estimators" : range(10,500,30),
    "max_features" : ['auto','sqrt','log2'],
    "max_depth" : [4,8,12],
    "min_samples_leaf" : [1,2,4]
    }
    gs_rf = GridSearchCV(estimator = regressor, param_grid = params, scoring = 'neg_mean_absolute_error', cv = 5, verbose = 4 )
    gs_rf.fit(X_train, y_train)
    return gs_rf.best_estimator_


def xg_boosting(X_train,y_train,X_test,y_test):
    model_XG = XGBRegressor()
    model_XG.fit(X_train, y_train)
    np.mean(cross_val_score(model_XG,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))
    model_xg = xg_tuning(X_train, y_train, model_XG)
    prediction_xg = model_xg.predict(X_test)
    result_xg = mean_absolute_error(y_test,prediction_xg)
    '''
    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, label="actual salary")
    plt.plot(x_ax, prediction_xg, label ="predicted salary")
    plt.title("test vs predicted")
    plt.title("XGBoost Regression")
    plt.legend()
    plt.show()
    '''
    return result_xg


def xg_tuning(X_train, y_train, model_XG):
    grid = {
        "learning_rate" : [0.001, 0.1, 0.1],
        "max_depth" : [3,6,9],
        "n_estimators" : range(10,500,30),
        "gamma" : [0.01, 0.1]
    }
    gs = GridSearchCV(estimator = model_XG, param_grid = grid, scoring = 'neg_mean_absolute_error', cv = 5, verbose = 4 )
    gs.fit(X_train, y_train)
    return gs.best_estimator_


def main(in_directory):
    df = pd.read_csv(in_directory)
    final_vars = df[['Rating','Industry','Size','Type of ownership','Sector','Revenue','total_comp','Per_Hour','EPS',
                 'state','same_loc','age','python','SQL',
               'aws','excel', 'jobs_cleaned', 'position_level', 'size_desc','average_salary']]
    cat_vars = pd.get_dummies(final_vars)
    X = cat_vars.drop('average_salary', axis = 1)
    y = cat_vars['average_salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    score_lm_val = linear_regression(X_train,y_train,X_test,y_test)
    score_rf_val = random_forest(X_train,y_train,X_test,y_test)
    score_xg_val = xg_boosting(X_train,y_train,X_test,y_test)
    print(OUTPUT_TEMPLATE.format(
        score_lm = score_lm_val,
        score_rf = score_rf_val,
        score_xg = score_xg_val,
    ))
if __name__=='__main__':
    in_directory = sys.argv[1]
    main(in_directory)


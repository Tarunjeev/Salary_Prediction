#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Lasso


def main_title(x):
    if 'data engineer' in x.lower():
        return 'data engineer'
    elif 'data scientist' in x.lower() in x.lower():
        return 'data scientist'
    elif 'analyst' in x.lower() in x.lower():
        return 'data analyst'
    elif 'machine learning' in x.lower():
        return 'machine learning engineer'
    elif 'manager' in x.lower():
        return 'manager'
    elif 'director' in x.lower():
        return 'director'
    else:
        return 'none'
    
def position(x):
    if 'sr' in x.lower() or 'senior' in x.lower() or 'lead' in x.lower() or 'principal' in x.lower():
        return 'senior'
    elif 'jr' in x.lower() or 'jr.' in x.lower():
        return 'junior'
    else:
        return 'none'
    
def main(in_directory):
    df = pd.read_csv(in_directory)
    df['size_desc']  = df['Job Description'].apply(lambda x: len(x))
    df['age'] = df['Founded'].apply(lambda x : x if x < 1 else 2022 - x)
    df['Per_Hour'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
    df['EPS'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer provided salary:' in x.lower() else 0)
    df = df[df['Salary Estimate'] != '-1']
    salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
    clean_sal = salary.apply(lambda x: x.replace('K','').replace('$',''))
    new_sal = clean_sal.apply(lambda x: x.lower().replace('per hour','').replace('employer provided salary:',''))
    df['min_salary'] = new_sal.apply(lambda x: int(x.split('-')[0]))
    df['max_salary'] = new_sal.apply(lambda x: int(x.split('-')[1]))
    df['average_salary'] = (df['min_salary']+df['max_salary'])/2
    df['comp_text'] = df.apply(lambda x: x['Company Name'] if x['Rating'] <0 else x['Company Name'][:-4], axis = 1)
    df['state'] = df['Location'].apply(lambda x: x.split(',')[1])
    df['same_loc'] = df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis = 1)
    df['python'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0) 
    df['Masters'] = df['Job Description'].apply(lambda x: 1 if 'master' in x.lower() else 0)
    df['R'] = df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() else 0)
    df['SQL'] = df['Job Description'].apply(lambda x: 1 if 'sql' in x.lower() else 0) 
    df['spark'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)
    df['aws'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)
    df['excel'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)
    df['jobs_cleaned'] = df['Job Title'].apply(main_title)
    df['position_level'] = df['Job Title'].apply(position)
    df['state']= df.state.apply(lambda x: x.strip() if x.strip().lower() != 'los angeles' else 'CA')
    df['total_comp'] = df['Competitors'].apply(lambda x: len(x.split(',')) if x != '-1' else 0)
    df['min_salary'] = df.apply(lambda x: x.min_salary*2 if x.Per_Hour ==1 else x.min_salary, axis =1)
    df['max_salary'] = df.apply(lambda x: x.max_salary*2 if x.Per_Hour ==1 else x.max_salary, axis =1)
    df = df.drop(['Unnamed: 0'], axis =1)
    df.to_csv('Cleaned_Data_final.csv', index=False)
if __name__=='__main__':
    in_directory = sys.argv[1]
    main(in_directory)


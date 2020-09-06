# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 13:46:00 2020

@author: Tpeng
"""

##### ML 1 Lab 1 #####

########## # Data Preprocessing # ##########

# Read in the 5 datasets and combine them into 1 

# Import Packages
import pandas as pd
from pandas import set_option
set_option('display.max_columns',400)
from pandas_profiling import ProfileReport
from pandas import DataFrame
from pandas import concat
import numpy as np
import copy as cp
import os
import glob
import matplotlib
import re
import datetime

# Set directory containing the datasets 
os.chdir('C:\\Users\\Tpeng\\OneDrive\\Documents\\SMU\\Term 3\\Machine Learning\\Lab1\\DatasetAndPreprocessing')
path = os.getcwd()
print(path)

#Read and Concat all data files (excluding the combined dataset created later stored here)
all_files = glob.glob(path + "/*.csv")
i = 0

for obs in all_files:
    if re.search('Combined_Dataset', obs):
        print(i)
        i += 1
        all_files.remove(obs)
    else:
        i += 1
        
property_data = []


# Create the combined dataset

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    property_data.append(df)

combineddf = pd.concat(property_data, axis=0, ignore_index=True, sort = False)

##### Output initial data descriptions, types, counts
combineddf.dtypes
# Print dataset header
combineddf.head()
# Get total number of observations and missing values
combineddf.count()
# Determine attribute data types
combineddf.info()


# Create and export the combined dataset
#combineddf.to_csv("Combined_Dataset.csv", sep = ',')








# Split variables into ordinal, continuous and categorical
ordinal_vars = ['rooms', 'bedrooms', 'bathrooms' ]
continuous_vars = ['lat', 'lon', 'surface_total', 'surface_covered', 'price']
categorical_vars = ['ad_type', 'l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'currency', 'price_period', 'property_type', 'operation_type']
string_vars = ['id', 'title', 'description']
time_vars = ['start_date', 'end_date', 'created_on']

# Cast attributes as specific data type and fill in missing values
# Replace missing values for each attribute set
combineddf = combineddf.replace(to_replace = "9999-12-31", value = np.nan)
combineddf.isna().sum()

# Create a new dataframe with all missing values as -1 (this is done to allow the datatypes to be changed)
combineddf = combineddf.replace(to_replace= np.nan, value = -1)
combineddf = combineddf.replace(to_replace = "9999-12-31", value = -1)

# Change data types
combineddf[ordinal_vars] = combineddf[ordinal_vars].astype(np.int64)
combineddf[continuous_vars] = combineddf[continuous_vars].astype(np.float64)
combineddf[categorical_vars] = combineddf[categorical_vars].astype('category')
combineddf[string_vars] = combineddf[string_vars].astype(str)
combineddf[time_vars] = pd.to_datetime(combineddf[time_vars].stack(), format = "%Y-%m-%d").unstack()



# loop through each categorical variable and display levels 
for var in categorical_vars:
    print(var, ' : ', combineddf[var].unique())
    
# Replace and combine equivalent factor levels and replace -1 with np.nan
combineddf = combineddf.replace(to_replace = -1, value = np.nan)
combineddf = combineddf.replace(to_replace = 'Estados Unidos de América', value = 'Estado Unidos')

# Output new dataset: missing values are nan
combineddf.to_csv("Combined_Dataset.csv", sep = ',')






# Output descriptive statistics
combineddf.describe()



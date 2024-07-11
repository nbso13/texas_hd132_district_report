## Author: Nicholas Ornstein
## Date: November 1st 2022
## Title: stripping files of personal data

import this
import pandas as pd
from shapely import geometry
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import And
import os
import time

# print(os.getcwd())
# defining parameters
file_dir = '../data/'
file_stem = r'fiftyplusdemscorefinal20220923-16686599681'
file_type = '.csv'
df_1 = pd.read_csv(file_dir+file_stem+file_type,  sep='\t', lineterminator='\r', encoding = 'utf-16le')

file_dir = '../data/'
file_stem = r'lessthanfiftydemscorefinal20220923-13552151697'
file_type = '.csv'
df_2 = pd.read_csv(file_dir+file_stem+file_type,  sep='\t', lineterminator='\r', encoding = 'utf-16le')

import pandas as pd

# Read the CSV file into a DataFrame

# Display the column names
print(df_1.columns)

df_1['Voter File VANID'] = np.arange(1, len(df_1)+1)

# Define the columns to be deleted
columns_to_delete = ['mAddress', 'mCity', 'mState', 'mZip5', 'mZip4',
     'Address', 'City', 'State', 'Zip5', 'Zip4', 'LastName',
       'FirstName', 'MiddleName', 'Suffix', 'PreferredEmail',
       'Preferred Phone', 'SpanishLanguagePrefe',]

# # Delete the specified columns
df_1.drop(columns=columns_to_delete, inplace=True)

# # Save the modified DataFrame to a new CSV file in the specified directory
output_path = '../data/fiftyplusdemscorefinal_scrambled.csv'
df_1.to_csv(output_path, index=False)

# # Read the CSV file into a DataFrame

# Display the column names
print(df_2.columns)

# # Define the columns to be deleted
# columns_to_delete = ['column1', 'column2', 'column3']

# # Delete the specified columns
df_2.drop(columns=columns_to_delete, inplace=True)

# # Save the modified DataFrame to a new CSV file in the specified directory
output_path = '../data/lessthanfiftydemscorefinal_scrambled.csv'
df_2.to_csv(output_path, index=False)

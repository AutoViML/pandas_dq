###########################################################################################
#Copyright 2023 Google LLC
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
###########################################################################################
##### pandas_dq is a simple transformer pipeline for all kinds of pandas pipelines. ######
###########################################################################################
# This library has two major modules. 
###########################################################################################
# The first module find_dq finds all the problems:
# It detects missing values and suggests to impute them with mean, median, mode, or a constant value.
# It identifies rare categories and suggests to group them into a single category or drop them.
# It finds infinite values and suggests to replace them with NaN or a large value.
# It detects mixed data types and suggests to convert them to a single type or split them into multiple columns.
# It detects outliers and suggests to remove them or use robust statistics.
# It detects high cardinality features and suggests to reduce them using encoding techniques or feature selection methods.
# It detects highly correlated features and suggests to drop one of them or use dimensionality reduction techniques.
# It detects duplicate rows and columns and suggests to drop them or keep only one copy .
# It detects skewed distributions and suggests to apply transformations or scaling techniques .
# It detects imbalanced classes and suggests to use resampling techniques or class weights .
# It detects feature leakage and suggests to avoid using features that are not available at prediction time .
############################################################################################
# The second module, Fix_DQ fixes all the data quality problems that find_dq finds.
############################################################################################
####### This Transformer was inspired by ChatGPT and Bard's answers when I was searching for
#######  a quick and dirty data cleaning library. Since they couldn't find me any good ones,
#######  I decided to create a simple quick and dirty data cleaning library using ChatGPT and Bard.
#######  I dedicate this library to all the 1000's of researchers who worked to create LLM's.
############################################################################################
# Define a function to print data cleaning suggestions
# Import pandas and numpy libraries
import pandas as pd
import numpy as np
import copy
import os
# Define a function to print data quality report and suggestions to clean data
def dq_report(data, target=None, csv_engine="pandas", verbose=0):
    """
    This is a data quality reporting tool that accepts any kind of file format as a filename or as a 
    pandas dataframe as input and returns a report highlighting potential data quality issues in it. 
    The function performs the following data quality checks. More will be added periodically.
     It detects missing values and suggests to impute them with mean, median,
      mode, or a constant value. It also identifies rare categories and suggests to group them
       into a single category or to drop them. 
       The function finds infinite values and suggests to replace them with NaN or a
        large value. It detects mixed data types and suggests to convert them 
        to a single type or split them into multiple columns.
         The function detects duplicate rows and columns, outliers in numeric columns,
          high cardinality features only in categorical columns, and 
          highly correlated features. 
    Finally, the function identifies if the problem is a classification problem or
     a regression problem and checks if there is class imbalanced or target leakage in the dataset.
    """
    if not verbose:
        print("This is a summary report. Change verbose to 1 to see more details on each DQ issue.")
    # Check if the input is a string or a dataframe
    if isinstance(data, str):
        # Get the file extension
        ext = os.path.splitext(data)[-1]
        # Load the file into a pandas dataframe based on the extension
        if ext == ".csv":
            print("If large dataset, we will randomly sample 100K rows to speed up processing...")
            if csv_engine == 'pandas':
                # Upload the data file into Pandas
                df = pd.read_csv(data)
            elif csv_engine == 'polars':
                # Upload the data file into Polars
                import polars as pl
                df = pl.read_csv(data)
            elif csv_engine == 'parquet':
                # Upload the data file into Parquet
                import pyarrow as pa
                df = pa.read_table(data)
            else :
                # print the pandas version
                if str(pd.__version__)[0] == '2':
                    print(f"pandas version={pd.__version__}. Hence using pyarrow backend.")
                    df = pd.read_csv(data, engine='pyarrow', dtype_backend='pyarrow')
                else:
                    print(f"pandas version={pd.__version__}. Hence using pandas backend.")
                    df = pd.read_csv(data)
        elif ext == ".parquet":
            df = pd.read_parquet(data)
        elif ext in [".feather", ".arrow", "ftr"]:
            df = pd.read_feather(data)
        else:
            print("Unsupported file format. Please use CSV, parquet, feather or arrow.")
            return data
        ######## This is to sample the data if it is too large ###
        if df.shape[0] >= 1000000:
            df = df.sample(100000)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        print("Invalid input. Please provide a string (filename) or a dataframe.")
        return

    # Drop duplicate rows
    dup_rows = df.duplicated().sum()
    if dup_rows > 0:
        print(f'Alert: Dropping {dup_rows} duplicate rows can sometimes cause column data types to change to object. Double-check!')
        df = df.drop_duplicates()
    
    # Drop duplicate columns
    dup_cols = df.columns[df.columns.duplicated()]
    if len(dup_cols) > 0:
        print(f'Alert: Dropping {len(dup_cols)} duplicate cols which can cause column data types to change to object. Double-check!')
        ### DO NOT MODIFY THIS LINE. TOOK A LONG TIME TO MAKE IT WORK!!!
        ###  THis is the only way that dropping duplicate columns works. This is not found anywhere!
        df = df.T[df.T.index.duplicated(keep='first')].T


    ### This is the column that lists our data quality issues
    new_col = 'DQ Issue'
    good_col = "The Good News"
    bad_col = "The Bad News"

    # Create an empty dataframe to store the data quality issues
    dq_df1 = pd.DataFrame(columns=[good_col, bad_col])
    dq_df1 = dq_df1.T
    dq_df1["first_comma"] = ""
    dq_df1[new_col] = ""

    # Create an empty dataframe to store the data quality issues
    data_types = pd.DataFrame(
        df.dtypes,
        columns=['Data Type']
    )

    missing_values = df.isnull().sum()
    missing_values_pct = ((df.isnull().sum()/df.shape[0])*100)
    missing_cols = missing_values[missing_values > 0].index.tolist()
    number_cols = df.select_dtypes(include=["integer", "float"]).columns.tolist() # Get numerical columns
    float_cols = df.select_dtypes(include=[ "float"]).columns.tolist() # Get float columns
    id_cols = []
    zero_var_cols = []

    missing_data = pd.DataFrame(
        missing_values_pct,
        columns=['Missing Values%']
    )
    unique_values = pd.DataFrame(
        columns=['Unique Values%']
    )
    for row in list(df.columns.values):
        if row in float_cols:
            unique_values.loc[row] = ["NA"]
        else:
            unique_values.loc[row] = [int(100*df[row].nunique()/df.shape[0])]
            if df[row].nunique() == df.shape[0]:
                id_cols.append(row)
            elif df[row].nunique() == 1:
                zero_var_cols.append(row)
        
    maximum_values = pd.DataFrame(
        columns=['Maximum Value']
    )
    minimum_values = pd.DataFrame(
        columns=['Minimum Value']
    )
    for row in list(df.columns.values):
        if row not in missing_cols:
            maximum_values.loc[row] = [df[row].max()]
        elif row in number_cols:
            maximum_values.loc[row] = [df[row].max()]
    for row in list(df.columns.values):
        if row not in missing_cols:
            minimum_values.loc[row] = [df[row].min()]
        elif row in number_cols:
            minimum_values.loc[row] = [df[row].min()]

    ### now generate the data quality starter dataframe
    dq_df2 = data_types.join(missing_data).join(unique_values).join(minimum_values).join(maximum_values)

    ### set up additional columns    
    dq_df2["first_comma"] = ""
    dq_df2[new_col] = f""
    
    #### This is the first thing you need to do ###############
    if dup_rows > 0:
        new_string =  f"There are {dup_rows} duplicate columns in the dataset. Keep only one copy of them."
        dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
        dq_df1.loc[bad_col,'first_comma'] = ', '
    else:
        new_string =  f"There are no duplicate rows in this dataset"
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '
    ### DO NOT CHANGE THE NEXT LINE. The logic for columns is different. 
    if len(dup_cols) > 0:
        new_string =  f"There are {len(dup_cols)} duplicate columns in the dataset. Keep only one copy of {dup_cols}."
        dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
        dq_df1.loc[bad_col,'first_comma'] = ', '
    else:
        new_string =  f"There are no duplicate columns in this datatset"
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '

    # Detect ID columns in dataset and recommend removing them
    if len(id_cols) > 0:
        new_string = f"There are ID columns in the dataset. Recommend removing them before modeling."
        dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
        dq_df1.loc[bad_col,'first_comma'] = ', '
        for col in id_cols:
            # Append a row to the dq_df1 with the column name and the issue only if the column has a missing value
            new_string = f"Possible ID colum: drop before modeling process."
            dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
            dq_df2.loc[col,'first_comma'] = ', '
    else:
        new_string = f"There are no ID columns in the dataset. So no ID columns to remove before modeling."
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '

    # Detect ID columns in dataset and recommend removing them
    if len(zero_var_cols) > 0:
        new_string = f"There are zero-variance columns in the dataset. Recommend removing them before modeling."
        dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
        dq_df1.loc[bad_col,'first_comma'] = ', '
        for col in zero_var_cols:
            # Append a row to the dq_df1 with the column name and the issue only if the column has a missing value
            new_string = f"Zero-variance colum: drop before modeling process."
            dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
            dq_df2.loc[col,'first_comma'] = ', '
    else:
        new_string = f"There are no zero-variance columns in the dataset. So no zero-variance columns to remove before modeling."
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '

    # Detect missing values and suggests to impute them with mean, median, mode, or a constant value123
    #missing_values = df.isnull().sum()
    #missing_cols = missing_values[missing_values > 0].index.tolist()
    if len(missing_cols) > 0:
        for col in missing_cols:
            # Append a row to the dq_df1 with the column name and the issue only if the column has a missing value
            if missing_values[col] > 0:
                new_string = f"{missing_values[col]} missing values. Impute them with mean, median, mode, or a constant value such as 123."
                dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
                dq_df2.loc[col,'first_comma'] = ', '
    else:
        new_string = f"There are no columns with missing values in the dataset"
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '
    

    # Identify rare categories and suggests to group them into a single category or drop them123
    rare_threshold = 0.05 # Define a threshold for rare categories
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist() # Get categorical columns
    rare_cat_cols = []
    if len(cat_cols) > 0:
        for col in cat_cols:
            value_counts = df[col].value_counts(normalize=True)
            rare_values = value_counts[value_counts < rare_threshold].index.tolist()
            if len(rare_values) > 0:
                rare_cat_cols.append(col)
                # Append a row to the dq_df2 with the column name and the issue
                if len(rare_values) <= 10:
                    new_string = f"{len(rare_values)} rare categories: {rare_values}. Group them into a single category or drop the categories."
                else:
                    new_string = f"{len(rare_values)} rare categories: Too many to list. Group them into a single category or drop the categories."
                dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
                dq_df2.loc[col,'first_comma'] = ', '
    else:
        new_string =  f"There are no categorical columns with rare categories (< {100*rare_threshold:.0f} percent) in this dataset"
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '


    # Find infinite values and suggests to replace them with NaN or a large value123
    inf_values = df.replace([np.inf, -np.inf], np.nan).isnull().sum() - missing_values
    inf_cols = inf_values[inf_values > 0].index.tolist()
    if len(inf_cols) > 0:
        new_string =  f"There are {len(inf_cols)} columns with infinite values in the dataset. Replace them with NaN or a finite value."
        dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
        dq_df1.loc[bad_col,'first_comma'] = ', '
        for col in inf_cols:
            if inf_values[col] > 0:
                new_string = f"{inf_values[col]} infinite values. Replace them with a finite value."
                dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
                dq_df2.loc[col,'first_comma'] = ', '
    else:
        new_string =  f"There are no columns with infinite values in this dataset "
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '

    # Detect mixed data types and suggests to convert them to a single type or split them into multiple columns123
    mixed_types = df.applymap(type).nunique() # Get the number of unique types in each column
    mixed_cols = mixed_types[mixed_types > 1].index.tolist() # Get the columns with more than one type
    if len(mixed_cols) > 0:
        new_string = f"There are {len(mixed_cols)} columns with mixed data types in the dataset. Convert them to a single type or split them into multiple columns."
        dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
        dq_df1.loc[bad_col,'first_comma'] = ', '
        for col in mixed_cols:
            if mixed_types[col] > 1:
                new_string = f"Mixed dtypes: has {mixed_types[col]} different data types: "
                for each_class in df[col].apply(type).unique():
                    if each_class == str:
                        new_string +=  f" object,"
                    elif each_class == int:
                        new_string +=  f" integer,"
                    elif each_class == float:
                        new_string +=  f" float,"
                dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
                dq_df2.loc[col,'first_comma'] = ', '
    else:
        new_string =  f"There are no columns with mixed (more than one) dataypes in this dataset"
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '

                
    # Detect outliers in numeric cols
    num_cols = df.select_dtypes(include=["int", "float"]).columns.tolist() # Get numerical columns
    if len(num_cols) > 0:
        first_time = True
        outlier_cols = []
        for col in num_cols:
            q1 = df[col].quantile(0.25) # Get the first quartile
            q3 = df[col].quantile(0.75) # Get the third quartile
            iqr = q3 - q1 # Get the interquartile range
            lower_bound = q1 - 1.5 * iqr # Get the lower bound
            upper_bound = q3 + 1.5 * iqr # Get the upper bound
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col] # Get the outliers
            if len(outliers) > 0:
                outlier_cols.append(col)
                if first_time:
                    new_string = f"There are {len(num_cols)} numerical columns, some with outliers. Remove them or use robust statistics."
                    dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
                    dq_df1.loc[bad_col,'first_comma'] = ', '
                    first_time =False
                ### check if there are outlier columns and print them ##
                new_string = f"has {len(outliers)} outliers greater than upper bound ({upper_bound}) or lower than lower bound({lower_bound}). Cap them or remove them."
                dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
                dq_df2.loc[col,'first_comma'] = ', '
        if len(outlier_cols) < 1:
            new_string =  f"There are no numeric columns with outliers in this dataset"
            dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
            dq_df1.loc[good_col,'first_comma'] = ', '
                
    # Detect high cardinality features only in categorical columns
    cardinality_threshold = 100 # Define a threshold for high cardinality
    cardinality = df[cat_cols].nunique() # Get the number of unique values in each categorical column
    high_card_cols = cardinality[cardinality > cardinality_threshold].index.tolist() # Get the columns with high cardinality
    if len(high_card_cols) > 0:
        new_string = f"There are {len(high_card_cols)} columns with high cardinality (>{cardinality_threshold} categories) in the dataset. Reduce them using encoding techniques or feature selection methods."
        dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
        dq_df1.loc[bad_col,'first_comma'] = ', '
        for col in high_card_cols:
            new_string = f"high cardinality with {cardinality[col]} unique values: Use hash encoding or embedding to reduce dimension."
            dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
            dq_df2.loc[col,'first_comma'] = ', '
    else:
        new_string =  f"There are no high cardinality columns in this dataset"
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '

    # Detect highly correlated features
    correlation_threshold = 0.8 # Define a threshold for high correlation
    correlation_matrix = df.corr().abs() # Get the absolute correlation matrix of numerical columns
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)) # Get the upper triangle of the matrix
    high_corr_cols = [column for column in upper_triangle.columns if any(upper_triangle[column] > correlation_threshold)] # Get the columns with high correlation
    if len(high_corr_cols) > 0:
        new_string = f"There are {len(high_corr_cols)} columns with >= {correlation_threshold} correlation in the dataset. Drop one of them or use dimensionality reduction techniques."
        dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
        dq_df1.loc[bad_col,'first_comma'] = ', '
        for col in high_corr_cols:
            new_string = f"has a high correlation with {upper_triangle[col][upper_triangle[col] > correlation_threshold].index.tolist()}. Consider dropping one of them."
            dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
            dq_df2.loc[col,'first_comma'] = ', '
    else:
        new_string =  f"There are no highly correlated columns in the dataset."
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '

    # First see if this is a classification problem 
    if target is not None:
        if isinstance(target, str):
            target_col = [target]
        else:
            target_col = copy.deepcopy(target) # Define the target column name
        
        cat_cols = df[target_col].select_dtypes(include=["object", "category"]).columns.tolist() 
        
        ### Check if it is a categorical var, then it is classification problem ###
        model_type = 'Regression'
        if len(cat_cols) > 0:
            model_type =  "Classification"
        else:
            int_cols = df[target_col].select_dtypes(include=["integer"]).columns.tolist() 
            copy_target_col = copy.deepcopy(target_col)
            for each_target_col in copy_target_col:
                if len(df[each_target_col].value_counts()) <= 30:
                    model_type =  "Classification"
        
        ### Then check for imbalanced classes in each target column
        if model_type == 'Classification':
            for each_target_col in target_col:
                y = df[each_target_col]
                # Get the value counts of each class
                value_counts = y.value_counts(normalize=True)
                # Get the minimum and maximum class frequencies
                min_freq = value_counts.min()
                max_freq = value_counts.max()
                # Define a threshold for imbalance
                imbalance_threshold = 0.1

                # Check if the class frequencies are imbalanced
                if min_freq < imbalance_threshold or max_freq > 1 - imbalance_threshold:
                    # Print a message to suggest resampling techniques or class weights
                    new_string =  f"Imbalanced classes in target variable ({each_target_col}). Use resampling or class weights to address."
                    dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
                    dq_df1.loc[bad_col,'first_comma'] = ', '
            
    # Detect target leakage in each feature
    if target is not None:
        target_col = copy.deepcopy(target) # Define the target column name
        if isinstance(target, str):
            preds = [x for x in list(df) if x not in [target_col]]
        else:
            preds = [x for x in list(df) if x not in target_col]
        leakage_threshold = 0.8 # Define a threshold for feature leakage
        leakage_matrix = df[preds].corrwith(df[target_col]).abs() # Get the absolute correlation matrix of each column with the target column
        leakage_cols = leakage_matrix[leakage_matrix > leakage_threshold].index.tolist() # Get the columns with feature leakage
        if len(leakage_cols) > 0:
            new_string = f"There are {len(leakage_cols)} columns with data leakage. Double check whether you should use this variable."
            dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
            dq_df1.loc[bad_col,'first_comma'] = ', '
            for col in leakage_cols:
                new_string = f"    {col} has a correlation >= {leakage_threshold} with {target_col}. Possible data leakage. Double check this variable."
                dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
                dq_df2.loc[col,'first_comma'] = ', '
        else:
            new_string =  f'There are no target leakage columns in the dataset'
            dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
            dq_df1.loc[good_col,'first_comma'] = ', '
    else:
        new_string = f'There is no target given. Hence no target leakage columns detected in the dataset'
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '

    dq_df1.drop('first_comma', axis=1, inplace=True)
    dq_df2.drop('first_comma', axis=1, inplace=True)
    for col in list(df):
        if dq_df2.loc[col, new_col] == "":
            dq_df2.loc[col,new_col] += "No issue"

    from IPython.display import display

    if verbose == 0:
        all_rows = dq_df1.shape[0]
        ax = dq_df1.head(all_rows).style.background_gradient(cmap='Reds').set_properties(**{'font-family': 'Segoe UI'})
        display(ax);

    if verbose >= 1:
        all_rows = dq_df2.shape[0]
        ax = dq_df2.head(all_rows).style.background_gradient(cmap='Reds').set_properties(**{'font-family': 'Segoe UI'})
        display(ax);

    # Return the dq_df1 as a table
    return dq_df2

##################################################################################################
# Import pandas and numpy libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import FunctionTransformer
import pdb

# Import BaseEstimator and TransformerMixin from sklearn
from sklearn.base import BaseEstimator, TransformerMixin

##################################################################################
def left_subtract(l1,l2):
    lst = []
    for i in l1:
        if i not in l2:
            lst.append(i)
    return lst
################################################################################

# Define a custom transformer class for fixing data quality issues
class Fix_DQ(BaseEstimator, TransformerMixin):
    # Initialize the class with optional parameters for the quantile, cat_fill_value and num_fill_value
    def __init__(self, quantile=0.75, cat_fill_value="missing", num_fill_value=9999, 
                 rare_threshold=0.05, correlation_threshold=0.8):
        self.quantile = quantile # Define a threshold for IQR for outlier detection 
        self.cat_fill_value = cat_fill_value ## Define a fill value for missing categories
        self.num_fill_value = num_fill_value # Define a fill value for missing numbers
        self.rare_threshold = rare_threshold # Define a threshold for rare categories
        self.correlation_threshold = correlation_threshold ## Above this limit, variables will be dropped
    
    # Define a function to cap the outliers in numerical columns using the upper bounds
    def cap_outliers(self, X):
        # Check if X is a pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            # Convert X to a pandas DataFrame
            X = pd.DataFrame(X)
            
        X = copy.deepcopy(X)
        
        # Get the numerical columns
        num_cols = X.select_dtypes(include=[ "float"]).columns.tolist()
        
        # Loop through each float column
        for col in num_cols:
            # Check if the column has an upper bound calculated in the fit method
            if col in self.upper_bounds_:
                # Cap the outliers using the upper bound
                X[col] = np.where(X[col] > self.upper_bounds_[col], self.upper_bounds_[col], X[col])
            else:
                # Just print a message and don't cap the outliers in that column
                print(f"No cap value found for column {col}. Continue...")                
        
        # Return the DataFrame with capped outliers
        return X
    
    # Define a function to impute the missing values in categorical and numerical columns using the constant values
    def impute_missing(self, X):
        """
        ### impute_missing can fill missing value using a global default value or a 
        ### dictionary of fill values for each column and apply that fill value to each column.
        """
        # Check if X is a pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            # Convert X to a pandas DataFrame
            X = pd.DataFrame(X)
            
        X = copy.deepcopy(X)

        # Get the categorical columns
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        
        # Get the numerical columns
        num_cols = X.select_dtypes(include=["int", "float"]).columns.tolist()
        
        # Impute the missing values in categorical columns with the cat_fill_value
        # Loop through the columns of cat_cols
        for col in cat_cols:
            # Check if the column is in the fill_values dictionary
            if isinstance(self.cat_fill_value, dict):
                if col in self.cat_fill_value:
                    # Impute the missing values in the column with the corresponding fill value
                    X[col] = X[[col]].fillna(self.cat_fill_value[col]).values
                else:
                    ### use a default value for that column since it is not specified
                    X[col] = X[[col]].fillna("missing").values
            else:
                ### use a global default value for all columns
                X[col] = X[[col]].fillna(self.cat_fill_value).values
        
        # Impute the missing values in numerical columns with the num_fill_value
        # Loop through the columns of num_cols
        for col in num_cols:
            # Check if the column is in the fill_values dictionary
            if isinstance(self.num_fill_value, dict):
                if col in self.num_fill_value:
                    # Impute the missing values in the column with the corresponding fill value
                    X[col] = X[[col]].fillna(self.num_fill_value[col]).values
                else:
                    ### use a default value for that column since it is not specified
                    X[col] = X[[col]].fillna(-999).values
            else:
                X[col] = X[[col]].fillna(self.num_fill_value).values
        
        # Return the DataFrame with imputed missing values
        return X
    
    # Define a function to identify rare categories and group them into a single category
    def group_rare_categories(self, X):
        # Check if X is a pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            # Convert X to a pandas DataFrame
            X = pd.DataFrame(X)
        
        # Get the categorical columns
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
                
        
        # Loop through each categorical column
        for col in cat_cols:
            # Get the value counts of each category
            value_counts = X[col].value_counts(normalize=True)
            # Get the rare categories that have a frequency below the threshold
            rare_values = value_counts[value_counts < self.rare_threshold].index.tolist()
            # Check if there are any rare categories
            if len(rare_values) > 0:
                # Group the rare categories into a single category called "Rare"
                X[col] = X[col].replace(rare_values, "Rare")
        
        # Return the DataFrame with grouped rare categories
        return X
    
    # Define a function to find infinite values and replace them with the upper bounds from that numeric column
    def replace_infinite(self, X):
        # Check if X is a pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            # Convert X to a pandas DataFrame
            X = pd.DataFrame(X)
        
        # Get the numerical columns
        num_cols = X.select_dtypes(include=["int", "float"]).columns.tolist()
        
        # Loop through each numerical column
        for col in num_cols:
            # Check if the column has an upper bound calculated in the fit method
            if col in self.upper_bounds_:
                # Replace the infinite values with the upper bound
                X[col] = X[col].replace([np.inf, -np.inf], self.upper_bounds_[col])
        
        # Return the DataFrame with replaced infinite values
        return X

    def detect_duplicates(self, X):
        # Check if X is a pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            # Convert X to a pandas DataFrame
            X = pd.DataFrame(X)
        
        # Drop duplicate rows
        dup_rows = X.duplicated().sum()
        if dup_rows > 0:
            print(f'Alert: Detecting {dup_rows} duplicate rows...')
        
        # Drop duplicate columns
        dup_cols = X.columns[X.columns.duplicated()]
        ### Remember that the logic for columns is different. Don't change this line!
        if len(dup_cols) > 0:
            print(f'Alert: Detecting {len(dup_cols)} duplicate cols...')
            ### DO NOT TOUCH THIS LINE!! IT TOOK A LONG TIME TO MAKE IT WORK!!
            ### Also if you don't delete these columns, then nothing after this line will work!
            X = X.T[X.T.index.duplicated(keep='first')].T
        
        # Return the DataFrame with no duplicates
        return X

    # Define a function to detect duplicate rows and columns and keep only one copy
    def drop_duplicated(self, X):
        # Check if X is a pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            # Convert X to a pandas DataFrame
            X = pd.DataFrame(X)
        
        # Drop duplicate rows
        dup_rows = X.duplicated().sum()
        if dup_rows > 0:
            print(f'Alert: Dropping {dup_rows} duplicate rows can sometimes cause column data types to change to object. Double-check!')
            X = X.drop_duplicates(keep='first')
        
        # Drop duplicate columns
        dup_cols = X.columns[X.columns.duplicated()]
        ### Remember that the logic for columns is different. Don't change this line!
        if len(dup_cols) > 0:
            print(f'Alert: Dropping {len(dup_cols)} duplicate cols: {dup_cols}!')
            ### DO NOT TOUCH THIS LINE!! IT TOOK A LONG TIME TO MAKE IT WORK!!
            X = X.T[X.T.index.duplicated(keep='first')].T
        
        # Return the DataFrame with no duplicates
        return X
    
    # Define a function to detect skewed distributions and apply a proper transformation to the column
    def transform_skewed(self, X):
        # Check if X is a pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            # Convert X to a pandas DataFrame
            X = pd.DataFrame(X)
        
        # Get the numerical columns
        num_cols = X.select_dtypes(include=["float"]).columns.tolist()
                
        # Loop through each numerical column
        for col in num_cols:
            # Find if a column transformer exists for this column
            if col in self.col_transformers_:
                # Cap the outliers using the upper bound
                if str(self.col_transformers_[col]).split("(")[0] == "PowerTransformer":
                    ### power transformer expects Pandas DataFrame
                    pt = self.col_transformers_[col]
                    X[col] = pt.transform(X[[col]])
                else:
                    ### function transformer expects pandas series
                    ft = self.col_transformers_[col]
                    X[col] = ft.transform(X[col])
        
        # Return the DataFrame with transformed skewed columns
        return X
    
    # Define the fit method that calculates the upper bound for each numerical column
    def fit(self, X, y=None):
        # Check if X is a pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            # Convert X to a pandas DataFrame
            X = pd.DataFrame(X)
        
        # Get the numerical columns
        num_cols = X.select_dtypes(include=["int", "float"]).columns.tolist()
        float_cols = X.select_dtypes(include=["float"]).columns.tolist()
        non_float_cols = left_subtract(X.columns, float_cols)

        ### First and foremost you must drop duplicate columns and rows
        X = self.detect_duplicates(X)

        # Detect ID columns
        self.id_cols_ = [column for column in non_float_cols if X[column].nunique() == X.shape[0]]
        if len(self.id_cols_) > 0:
            print(f"{len(self.id_cols_)} ID cols will be dropped from further processing: {self.id_cols_}")

        # Detect zero-variance columns
        self.zero_var_cols_ = [column for column in non_float_cols if X[column].nunique() == 1]
        if len(self.zero_var_cols_) > 0:
            print(f"    {len(self.zero_var_cols_)} zero-variance cols will be dropped from further processing: {self.zero_var_cols_}")
        
        # Detect highly correlated features
        self.drop_corr_cols_ = []
        correlation_matrix = X.corr().abs() # Get the absolute correlation matrix of numerical columns
        upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)) # Get the upper triangle of the matrix
        high_corr_cols = [column for column in upper_triangle.columns if any(upper_triangle[column] > self.correlation_threshold)] # Get the columns with high correlation
        if len(high_corr_cols) > 0:
            self.drop_corr_cols_ = high_corr_cols
            for col in high_corr_cols:
                    print(f"    Dropping {col} which has a high correlation with {upper_triangle[col][upper_triangle[col] > self.correlation_threshold].index.tolist()}")
        
        
        # Initialize an empty dictionary to store the upper bounds
        self.upper_bounds_ = {}
        
        # Loop through each numerical column
        #### processing of quantiles is only for float columns or those in dict ###
        if self.quantile is None:
            ### you still need to calculate upper bounds needed capping for infinite values ##
            base_quantile = 0.75
            for col in float_cols:
                # Get the third quartile
                q3 = X[col].quantile(base_quantile)
                # Get the interquartile range
                iqr = X[col].quantile(base_quantile) - X[col].quantile(1 - base_quantile)
                # Calculate the upper bound
                upper_bound = q3 + 1.5 * iqr
                # Store the upper bound in the dictionary
                self.upper_bounds_[col] = upper_bound
        else:
            ### calculate upper bounds to cap outliers using quantile given ##
            for col in float_cols:
                # Get the third quartile
                q3 = X[col].quantile(self.quantile)
                # Get the interquartile range
                iqr = X[col].quantile(self.quantile) - X[col].quantile(1 - self.quantile)
                # Calculate the upper bound
                upper_bound = q3 + 1.5 * iqr
                # Store the upper bound in the dictionary
                self.upper_bounds_[col] = upper_bound

        # Initialize an empty dictionary to store the column transformers
        self.col_transformers_ = {}
        
        # Define a threshold for skewness
        skew_threshold = 1.0
        
        # Loop through each float column
        for col in float_cols:
            # Calculate the skewness of the column
            skewness = X[col].skew()
            # Check if the skewness is above the threshold
            if abs(skewness) > skew_threshold:
                # Apply a log transformation if the column has positive values only
                if X[col].min() > 0:
                    ### function transformer expects pandas series
                    ft = FunctionTransformer(np.log1p)
                    ft.fit(X[col])
                    self.col_transformers_[col] = ft
                # Apply a box-cox transformation if the column has positive values only and scipy is installed
                elif X[col].min() > 0 and "scipy" in sys.modules:
                    ### power transformer expects Pandas DataFrame
                    pt = PowerTransformer(method="box-cox")
                    pt.fit(X[[col]])
                    self.col_transformers_[col] = pt
                # Apply a yeo-johnson transformation if the column has any values and sklearn is installed
                else:
                    ### power transformer expects Pandas DataFrame
                    pt = PowerTransformer(method="yeo-johnson")
                    pt.fit(X[[col]])
                    self.col_transformers_[col] = pt

        # Get the number of unique types in each column
        self.mixed_type_cols_ = []
        mixed_types = X.applymap(type).nunique()
        # Get the columns with more than one type
        self.mixed_type_cols_ = mixed_types[mixed_types > 1].index.tolist()
        if len(self.mixed_type_cols_) > 0:
            print(f"    {len(self.mixed_type_cols_)} mixed data type cols will be dropped from further processing: {self.mixed_type_cols_}")
                
        # Return the fitted transformer object
        return self
    
    # Define the transform method that calls the cap_outliers and impute_missing functions on the input DataFrame
    def transform(self, X):
        # Check if X is a pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            # Convert X to a pandas DataFrame
            X = pd.DataFrame(X)

        # First find duplicate columns and rows
        X = self.drop_duplicated(X)

        
        ### drop mixed data type columns from further processing ##
        if len(self.id_cols_) > 0:
            X = X.drop(self.id_cols_, axis=1)

        ### drop mixed data type columns from further processing ##
        if len(self.zero_var_cols_) > 0:
            X = X.drop(self.zero_var_cols_, axis=1)

        ### drop mixed data type columns from further processing ##
        if len(self.mixed_type_cols_) > 0:
            drop_cols = left_subtract(self.mixed_type_cols_, self.zero_var_cols_+self.id_cols_)
            if len(drop_cols) > 0:
                X = X.drop(drop_cols, axis=1)
            else:
                drop_cols = left_subtract(self.zero_var_cols_+self.id_cols_, self.mixed_type_cols_)
                if len(drop_cols) > 0:
                    X = X.drop(drop_cols, axis=1)
            
        ### drop highly correlated columns from further processing ##
        if len(self.drop_corr_cols_) > 0:
            if len(left_subtract(self.drop_corr_cols_,self.mixed_type_cols_)) > 0:
                extra_cols = left_subtract(self.drop_corr_cols_,self.mixed_type_cols_)
            elif len(left_subtract(self.mixed_type_cols_,drop_corr_cols_)) > 0:
                extra_cols = left_subtract(self.mixed_type_cols_, self.drop_corr_cols_)
            if len(extra_cols) > 0:
                X = X.drop(extra_cols, axis=1)
            
        
        # Call the impute_missing function first and assign it to a new variable 
        imputed_X = self.impute_missing(X)

        if self.quantile is None:
            #### Don't do any processing if quantile is set to None ###
            capped_X = copy.deepcopy(imputed_X)
        else:
            # Call the cap_outliers function on X and assign it to a new variable 
            capped_X = self.cap_outliers(imputed_X)
        
        # Call the replace_infinite function on capped_X and assign it to a new variable 
        infinite_X = self.replace_infinite(capped_X)
        
        # Call the group_rare_categories function on infinite_X and assign it to a new variable 
        rare_X = self.group_rare_categories(infinite_X)
        
        # Call the power transformer function on rare_X and assign it to a new variable 
        transformed_X = self.transform_skewed(rare_X)
                
        # Return the transformed DataFrame
        return transformed_X
################################################################################
from IPython.display import display
# Import BaseEstimator and TransformerMixin from sklearn
from sklearn.base import BaseEstimator, TransformerMixin
import copy
class DataSchemaChecker(BaseEstimator, TransformerMixin):
    """
    A class to check if a pandas dataframe conforms to a given schema.

    Attributes:
        schema (dict): A dictionary that maps column names to data types.
    
    Example of a schema: all python dtypes must be surrounded by quote strings.
    {'name': 'string',
     'age': 'float32',
     'gender': 'object',
     'income': 'float64',
     'target': 'integer'}

    Methods:
        fit(df): Checks if the dataframe matches the schema and prints a table of errors if any.
    """
    def __init__(self, schema):
        """
        Initializes the DataFrameSchemaChecker object with a schema.

        Args:
            schema (dict): A dictionary that maps column names to data types.
            
        Example of a schema: all python dtypes must be surrounded by quote strings.
        {'name': 'string',
         'age': 'float32',
         'gender': 'object',
         'income': 'float64',
         'target': 'integer'}
        """
        self.schema = schema

    def fit(self, df):
        """
        Checks if the dataframe matches the schema and prints a table of errors if any.

        Args:
            df (pd.DataFrame): The dataframe to be checked.

        Raises:
            ValueError: If the number of columns in the dataframe does not match the number of columns in the schema or if the schema contains an invalid data type.

        Returns:
            None
        """
        # Check if the number of columns in the dataframe matches the number of columns in the schema
        if len(df.columns) != len(self.schema):
            raise ValueError("The number of columns in the dataframe does not match the number of columns in the schema")

        # Translate the schema to pandas dtypes
        translated_schema = {}
        for column, dtype in self.schema.items():
            if dtype in ["string","object","category", "str"]:
                translated_schema[column] = "object"
            elif dtype in ["text","NLP","nlp"]:
                translated_schema[column] = "object"
            elif dtype in ["boolean","bool"]:
                translated_schema[column] = "bool"
            elif dtype in [ "np.int8", "int8"]:
                translated_schema[column] = "int8"
            elif dtype in ["np.int16","int16"]:
                translated_schema[column] = "int16"
            elif dtype in ["int32", "np.int32"]:
                translated_schema[column] = "int32"
            elif dtype in ["integer","int", "int64", "np.int64"]:
                translated_schema[column] = "int64"
            elif dtype in ["date"]:
                translated_schema[column] = "datetime64[ns]"                
            elif dtype in ["float"]:
                translated_schema[column] = "float64"
            elif dtype in ["np.float32", "float32"]: 
                translated_schema[column] = "float32"
            elif dtype in ["np.float64", "float64"]:
                translated_schema[column] = "float64"
            else:
                raise ValueError("Invalid data type: {}".format(dtype))

        # Check if the data types of the columns in the dataframe match the data types in the schema
        errors = []
        for column, dtype in translated_schema.items():
            # Get the actual data type of the column
            actual_dtype = df[column].dtype
            # Compare with the expected data type
            if actual_dtype != dtype:
                # Append an error message to the list
                errors.append({
                    "column": column,
                    "expected_dtype": dtype,
                    "actual_dtype": actual_dtype,
                    "data_dtype_mismatch": "Column '{}' has data type '{}' but expected '{}'".format(
                        column, actual_dtype, dtype)})
        
        # Print a table of errors if there are any
        if errors:
            # Create a dataframe from the list of errors
            self.error_df_ = pd.DataFrame(errors)
            # Display the dataframe using IPython.display
            display(self.error_df_)
        else:
            print("**No Data Schema Errors**")

        return self
            
    def transform(self, df):
        """
        Transforms the dataframe dtype to the expected dtype if the dataframe has been fit with DataSchemaChecker

        Args:
            df (pd.DataFrame): The dataframe to be transformed using DataSchemaChecker's error_df_ variable.

        Raises:
            Error: If the datatype for a column cannot be transformed as requested.

        Returns:
            modified dataframe (df)
        """        
        df = copy.deepcopy(df)
        if len(self.error_df_) > 0:
            # Loop over only the error data types detected in the Error DataFrame.
            for i, row in self.error_df_.iterrows():
                column = row['column']
                try:
                    if row['expected_dtype']=='datetime64[ns]':
                        df[column] = pd.to_datetime(df[column])
                    else:
                        df[column] = df[column].astype(row["expected_dtype"])
                except:
                    print(f"Converting {column} to {self.error_df_['expected_dtype'][0]} is erroring. Please convert it yourself.")
                
        return df

############################################################################################
module_type = 'Running' if  __name__ == "__main__" else 'Imported'
version_number =  '1.10'
print(f"""{module_type} pandas_dq ({version_number}). Always upgrade to get latest version.
""")
#################################################################################

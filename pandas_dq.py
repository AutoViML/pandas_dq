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
# The first module dq_report finds all the problems:
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
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('colheader_justify', 'center')
# Define a function to print a data quality report and suggestions to clean data
def dq_report(data, target=None, html=False, csv_engine="pandas", verbose=0):
    """
    This is a data quality reporting tool that accepts any kind of file format as a filename or as a 
    pandas dataframe as input and returns a report highlighting potential data quality issues in it. 
    The function performs the following data quality checks. More will be added periodically.
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
    Finally, the function identifies if the dataset is a classification problem or
     a regression problem and checks if there is class imbalance or target leakage in the dataset.
    """

    correlation_threshold = 0.8 # Define a threshold for high correlation between variables
    leakage_threshold = 0.8 # Define a threshold for feature leakage
    if not verbose:
        print("This is a summary report. Change verbose to 1 to see more details on each DQ issue.")
    #### If sometimes, target is given as empty string, change it to None
    if isinstance(target, str):
        if target == '':
            target = None        
    # Check if the input is a string or a dataframe
    if isinstance(data, str):
        # Get the file extension
        ext = os.path.splitext(data)[-1]
        # Load the file into a pandas dataframe based on the extension
        if ext == ".csv":
            print("    If large dataset, we will randomly sample 100K rows to speed up processing...")
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
                    print(f"    pandas version={pd.__version__}. Hence using pyarrow backend.")
                    df = pd.read_csv(data, engine='pyarrow', dtype_backend='pyarrow')
                else:
                    print(f"    pandas version={pd.__version__}. Hence using pandas backend.")
                    df = pd.read_csv(data)
        elif ext == ".parquet":
            df = pd.read_parquet(data)
        elif ext in [".feather", ".arrow", "ftr"]:
            df = pd.read_feather(data)
        else:
            print("    Unsupported file format. Please use CSV, parquet, feather or arrow.")
            return data
        ######## This is to sample the data if it is too large ###
        if df.shape[0] >= 1000000:
            df = df.sample(100000)
    elif isinstance(data, pd.DataFrame):
        df = copy.deepcopy(data)
    else:
        print("    Unrecognized input. Please provide a filename or a pandas dataframe. Returning...")
        return data

    # Drop duplicate rows
    dup_rows = df.duplicated().sum()
    if dup_rows > 0:
        print(f'There are {dup_rows} duplicate rows in your dataset')
        print(f'    Alert: Dropping duplicate rows can sometimes cause your column data types to change to object!')
        df = df.drop_duplicates()
    
    # Drop duplicate columns
    dup_cols = df.columns[df.columns.duplicated()]
    if len(dup_cols) > 0:
        print(f'    Alert: Dropping {len(dup_cols)} duplicate cols')
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
    #####       Classify Columns   ################
    if not target is None:
        var_df = classify_columns(df.drop(target, axis=1), verbose=0)
    else:
        var_df = classify_columns(df, verbose=0) 
    #####       ClassifyColumns   ################
    IDcols = var_df['id_vars']
    nlp_vars = var_df['nlp_vars']
    discrete_string_vars = var_df['discrete_string_vars']
    cols_delete = var_df['cols_delete']
    bool_vars = var_df['string_bool_vars'] + var_df['num_bool_vars']
    int_vars = var_df['int_vars']
    categorical_vars = var_df['cat_vars'] + var_df['factor_vars'] 
    date_vars = var_df['date_vars']
    if target is None:
        preds = [x for x in list(df) if x not in IDcols+cols_delete]
    else:
        if isinstance(target, str):
            ### target is a string #####
            preds = [x for x in list(df) if x not in IDcols+cols_delete+[target]]
        else:
            ### target is a multi-label list ####
            preds = [x for x in list(df) if x not in IDcols+cols_delete+target]
    #####################################################################################################        
    float_cols = var_df['continuous_vars'] # Get float columns
    id_cols = list(set(IDcols[:]))
    zero_var_cols = list(set(cols_delete[:]))
    number_cols = list(set(var_df['continuous_vars'] + var_df['int_vars']))
    text_vars = list(set(discrete_string_vars + nlp_vars))
    cat_cols = categorical_vars[:] # Get categorical columns
    date_cols = date_vars[:]
    #########################   These are needed for further processing ##########

    missing_data = pd.DataFrame(
        missing_values_pct,
        columns=['Missing Values%']
    )
    unique_values = pd.DataFrame(
        columns=['Unique Values%']
    )
    #### For every column except float columns find the number of unique values % ##
    for col in list(df.columns.values):
        if col in float_cols:
            unique_values.loc[col] = ["NA"]
        else:
            unique_values.loc[col] = [int(100*df[col].nunique()/df.shape[0])]


    #### Find the max and min of every column except missing cols ##        
    maximum_values = pd.DataFrame(
        columns=['Maximum Value']
    )
    minimum_values = pd.DataFrame(
        columns=['Minimum Value']
    )
    ### for every column except missing cols, find the max and min ######
    for col in list(df.columns.values):
        if col not in missing_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                maximum_values.loc[col] = [df[col].max()]
        elif col in number_cols:
            maximum_values.loc[col] = [df[col].max()]

    ### for every column except missing cols, find the max and min ######
    for col in list(df.columns.values):
        if col not in missing_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                minimum_values.loc[col] = [df[col].min()]
        elif col in number_cols:
            minimum_values.loc[col] = [df[col].min()]
    
    ### now generate the data quality starter dataframe
    dq_df2 = data_types.join(missing_data).join(unique_values).join(minimum_values).join(maximum_values)
    dq_df2['Minimum Value'] = dq_df2[['Minimum Value']].fillna('') ## these need to be filled with empty strings
    dq_df2['Maximum Value'] = dq_df2[['Maximum Value']].fillna('') ### these need to be filled with empty strings

    ### set up additional columns    
    dq_df2["first_comma"] = ""
    dq_df2[new_col] = f""
    
    #### This is the first thing you need to do ###############
    if dup_rows > 0:
        new_string =  f"There are {dup_rows} duplicate rows in the dataset. De-Dup these rows using Fix_DQ."
        dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
        dq_df1.loc[bad_col,'first_comma'] = ', '
    else:
        new_string =  f"There are no duplicate rows in this dataset"
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '
    ### DO NOT CHANGE THE NEXT LINE. The logic for columns is different. 
    if len(dup_cols) > 0:
        new_string =  f"There are {len(dup_cols)} duplicate columns in the dataset. De-Dup {dup_cols} using Fix_DQ."
        dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
        dq_df1.loc[bad_col,'first_comma'] = ', '
    else:
        new_string =  f"There are no duplicate columns in this datatset"
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '

    # Detect ID columns in dataset and recommend removing them
    if len(id_cols) > 0:
        new_string = f"There are ID columns in the dataset. Remove them before modeling using Fix_DQ."
        dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
        dq_df1.loc[bad_col,'first_comma'] = ', '
        for col in id_cols:
            # Append a row to the dq_df1 with the column name and the issue only if the column has a missing value
            new_string = f"Possible ID column: drop before modeling step."
            dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
            dq_df2.loc[col,'first_comma'] = ', '
    else:
        new_string = f"There are no ID columns in the dataset."
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '

    # Detect zero variance columns in dataset and recommend removing them
    if len(zero_var_cols) > 0:
        new_string = f"These are zero-variance or low information columns in the dataset. Remove them before modeling."
        dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
        dq_df1.loc[bad_col,'first_comma'] = ', '
        for col in zero_var_cols:
            # Append a row to the dq_df1 with the column name and the issue only if the column has a missing value
            new_string = f"Possible Zero-variance or low information colum: drop before modeling step."
            dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
            dq_df2.loc[col,'first_comma'] = ', '
    else:
        new_string = f"There are no zero-variance or low information columns in the dataset."
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '

    # Detect date-time related columns in dataset 
    if len(date_cols) > 0:
        new_string =  f"There are {len(date_vars)} date-time vars in the dataset. Make sure you transform them before modeling."
        dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
        dq_df1.loc[bad_col,'first_comma'] = ', '
        for col in date_cols:
            # Append a row to the dq_df1 with the column name and the issue only if the column has a missing value
            new_string = f"Possible date-time colum: transform before modeling step."
            dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
            dq_df2.loc[col,'first_comma'] = ', '
    else:
        new_string =  f"There are no date-time vars in this dataset"
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
    rare_threshold = 0.01 # Define a 1% threshold for rare categories
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
    mixed_types = df[preds].applymap(type).nunique() # Get the number of unique types in each column
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
    num_cols = var_df['continuous_vars'] + var_df['int_vars'] # Get numerical columns
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
                new_string = f"Column has {len(outliers)} outliers greater than upper bound ({upper_bound:.2f}) or lower than lower bound({lower_bound:.2f}). Cap them or remove them."
                dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
                dq_df2.loc[col,'first_comma'] = ', '
        if len(outlier_cols) < 1:
            new_string =  f"There are no numeric columns with outliers in this dataset"
            dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
            dq_df1.loc[good_col,'first_comma'] = ', '
                
    # Detect high cardinality features only in categorical columns
    cardinality = df[discrete_string_vars].nunique() # Get the number of unique values in each categorical column
    cardinality_threshold = min(30, cardinality.min()) # Define a threshold for high cardinality
    high_card_cols = discrete_string_vars[:] 
    # Get the columns with high cardinality
    ## high_card_cols = cardinality[cardinality > cardinality_threshold].index.tolist() 
    if len(high_card_cols) > 0:
        new_string = f"There are {len(high_card_cols)} columns with high cardinality (>{cardinality_threshold} categories) in the dataset. Reduce them using encoding techniques or feature selection methods."
        dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
        dq_df1.loc[bad_col,'first_comma'] = ', '
        for col in high_card_cols:
            new_string = f"Possible high cardinality column with {cardinality[col]} unique values: Use hash encoding or text embedding to reduce dimension."
            dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
            dq_df2.loc[col,'first_comma'] = ', '
    else:
        new_string =  f"There are no high cardinality columns in this dataset"
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '

    # Detect highly correlated features
    correlation_matrix = df[num_cols].corr().abs() # Get the absolute correlation matrix of numerical columns
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)) # Get the upper triangle of the matrix
    high_corr_cols = [column for column in upper_triangle.columns if any(upper_triangle[column] > correlation_threshold)] # Get the columns with high correlation
    if len(high_corr_cols) > 0:
        new_string = f"There are {len(high_corr_cols)} columns with >= {correlation_threshold} correlation in the dataset. Drop one of them or use dimensionality reduction techniques."
        dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
        dq_df1.loc[bad_col,'first_comma'] = ', '
        for col in high_corr_cols:
            new_string = f"Column has a high correlation with {upper_triangle[col][upper_triangle[col] > correlation_threshold].index.tolist()}. Consider dropping one of them."
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
                    dq_df2.loc[each_target_col, new_col] += "Target column. Appears to have Imbalanced classes. Try balancing classes."
            
        # Detect target leakage in each feature
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
        target_col = []

    dq_df1.drop('first_comma', axis=1, inplace=True)
    dq_df2.drop('first_comma', axis=1, inplace=True)
    for col in list(df):
        if dq_df2.loc[col, new_col] == "":
            if col in target_col:
                #### This is to make sure target column is properly labeled.
                if df[col].nunique() == 1:
                    dq_df2.loc[col,new_col] += "Target column. Appears to have zero variance. Double-check it."
                else:
                    dq_df2.loc[col,new_col] += "Target column"
            else:
                dq_df2.loc[col,new_col] += "No issue"

    if html:
        if verbose == 0:
            write_to_html(dq_df1, "dq_report.html")
        else:
            write_to_html(dq_df2, "dq_report.html")
    else:
        try:
            from IPython.display import display
        except Exception as e:
            print('Erroring due to %s. Please install and try again...')
            return dq_df2
        if verbose < 0:
            pass
        elif verbose == 0:
            all_rows = dq_df1.shape[0]
            ax = dq_df1.head(all_rows).style.background_gradient(cmap='Reds').set_properties(**{'font-family': 'Segoe UI'})
            display(ax);
        else:
            all_rows = dq_df2.shape[0]
            ax = dq_df2.head(all_rows).style.background_gradient(cmap='Reds').set_properties(**{'font-family': 'Segoe UI'})
            display(ax);

        # Return the dq_df1 as a table
    return dq_df2
##################################################################################################
import re
# Import the webbrowser module
import webbrowser
def write_to_html(dqr, filename="dq_report.html"):
    """
    Write a data quality report to an HTML file and open it in a browser.

    Parameters
    ----------
    dqr : pandas.DataFrame
        A data quality report generated by the dq_report function.

    Returns
    -------
    None

    Notes
    -----
    This function will create an HTML file named "dq_report.html" in the current working directory 
    and open it in a new tab of the default browser. The HTML file will contain a table with 
    the data quality report, formatted with colors, fonts, and styles. The table will have 
    alternating row colors using the CSS style sheet embedded in strings. The function requires 
    the re and webbrowser modules to be imported.
    """
    df_html = dqr.to_html(classes="table table-striped table-bordered table-hover",
                border=0, na_rep="", index=True).replace('<th>', 
                '<th style="background-color: lightgreen">').replace('<td>', 
                '<td style="color: blue">')

    df_html = f""" <style> /* Import Roboto from Google Fonts */ @import url(‘https://fonts.googleapis.com/css?family=Roboto&display=swap’);

    /* Set the font family and size for the whole table */ table {{ font-family: Roboto; font-size: 12px; }}

    /* Set the background color and text alignment for the header cells */ th {{ background-color: orange; font-size: 14px; text-align: center; }}

    /* Set the color and font style for the data cells */ td {{ color: blue; font-style: italic; text-align: left; }}

    /* Set the background color for every odd row */ tr:nth-child(odd) {{ background-color: lightyellow; }}

    /* Set the background color for every even row */ tr:nth-child(even) {{ background-color: lightgrey; }} </style> {df_html} """

    # Return the HTML code of the report as a string
    with open(filename, "w") as f:
        f.write(df_html)

    # Open the file in a new tab of the default browser
    webbrowser.open_new_tab(filename)

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
def compare_unique(df1, df2, column):
    """
    Compare the difference between the unique values in a single column of two dataframes.

    This function takes two dataframes and a column name as inputs and returns a dictionary
    with the count and the actual differences of the unique values in that column between
    the two dataframes.

    Parameters
    ----------
    df1 : pandas.DataFrame
        The first dataframe to compare.
    df2 : pandas.DataFrame
        The second dataframe to compare.
    column : str
        The name of the column to compare.

    Returns
    -------
    result : dict
        A dictionary with four keys: 'count_1', 'count_2', 'diff_1_2', and 'diff_2_1'.
        'count_1' is the number of unique values in column of df1.
        'count_2' is the number of unique values in column of df2.
        'diff_1_2' is a list of unique values in column of df1 that are not in column of df2.
        'diff_2_1' is a list of unique values in column of df2 that are not in column of df1.
    """
    # Get the unique values in column of each dataframe as sets
    set1 = set(df1[column].unique())
    set2 = set(df2[column].unique())
    
    # Calculate the count and the differences using set operations
    count_1 = len(set1)
    count_2 = len(set2)
    diff_1_2 = list(set1 - set2)
    diff_2_1 = list(set2 - set1)
    
    # Store the results in a dictionary
    result = {
        "unique_count_in_df1": count_1,
        "unique_count_in_df2": count_2,
        "diff_between_df1_df2": diff_1_2,
        "diff_between_df2_df1": diff_2_1,
    }
    
    # Return the result
    return result
########################################################################################
# Define a custom transformer class for fixing data quality issues
class Fix_DQ(BaseEstimator, TransformerMixin):
    # Initialize the class with optional parameters for the quantile, cat_fill_value and num_fill_value
    def __init__(self, quantile=0.87, cat_fill_value="missing", num_fill_value=9999, 
                 rare_threshold=0.01, correlation_threshold=0.9):
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
                # Just continue and don't cap the outliers in that column
                continue
        
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
        
        missing_values = X.isnull().sum()
        missing_cols = missing_values[missing_values > 0].index.tolist()
        #### Sometimes missing values are found in test but not train. This to catch those!
        for col in missing_cols:
            if not col in self.missing_cols_:
                self.missing_cols_.append(col)

        # Loop through the columns of cat_cols
        for col in self.missing_cols_:
            if col in cat_cols:
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
        for col in self.missing_cols_:
            if col in num_cols:
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
        self.drop_cols_ = []
        self.missing_cols_ = []

        # Check if X is a pandas DataFrame        
        if not isinstance(X, pd.DataFrame):
            # Convert X to a pandas DataFrame
            X = pd.DataFrame(X)
        
        # Get the numerical columns
        num_cols = X.select_dtypes(include=["int", "float"]).columns.tolist()
        float_cols = X.select_dtypes(include=["float"]).columns.tolist()
        non_float_cols = left_subtract(X.columns, float_cols)

        # Find percent of missing values in all columns
        missing_values = X.isnull().sum()
        self.missing_cols_ = missing_values[missing_values > 0].index.tolist()
        drop_missing = []
        for each in self.missing_cols_:
            if X[each].isna().sum()/len(X) >= 0.80 :
                ### drop the column if it has 80% or more missing values
                drop_missing.append(each)
                print(f"    Dropping {each} since it has >= 80%% missing values")

        ### First and foremost you must drop duplicate columns and rows
        X = self.detect_duplicates(X)

        # Detect ID columns
        self.id_cols_ = [column for column in non_float_cols if X[column].nunique() == X.shape[0]]
        if len(self.id_cols_) > 0:
            print(f"    Dropping {len(self.id_cols_)} ID column(s): {self.id_cols_}")

        # Detect zero-variance columns
        self.zero_var_cols_ = [column for column in non_float_cols if X[column].nunique() == 1]
        if len(self.zero_var_cols_) > 0:
            print(f"    Dropping {len(self.zero_var_cols_)} zero-variance cols: {self.zero_var_cols_}")
        
        # Detect highly correlated features
        self.drop_corr_cols_ = []
        correlation_matrix = X[num_cols].corr().abs() # Get the absolute correlation matrix of numerical columns
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
            base_quantile = 0.99
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
            extra_mixed = left_subtract(self.mixed_type_cols_, self.missing_cols_)
            if len(extra_mixed) > 0:
                print(f"    Dropping {len(extra_mixed)} columns due to mixed data types")
                for each in extra_mixed:
                    print(f"        {each} has mixed dtypes: {X[each].apply(type).unique()}")    

        ### drop ID columns from further processing ##
        if len(self.id_cols_) > 0:
            self.drop_cols_ += self.id_cols_

        ### drop Zero Variance columns from further processing ##
        if len(self.zero_var_cols_) > 0:
            self.drop_cols_ += self.zero_var_cols_

        ### drop mixed data type columns from further processing ##
        if len(self.mixed_type_cols_) > 0:
            drop_cols = left_subtract(extra_mixed, self.zero_var_cols_+self.id_cols_)
            if len(drop_cols) > 0:
                self.drop_cols_ += drop_cols
            if len(extra_mixed) > 0:
                self.drop_cols_ += extra_mixed
            
        ### drop highly correlated columns from further processing ##
        if len(self.drop_corr_cols_) > 0:
            if len(left_subtract(self.drop_corr_cols_, self.drop_cols_)) > 0:
                extra_cols = left_subtract(self.drop_corr_cols_,self.drop_cols_)
                self.drop_cols_ += extra_cols

        ### drop columns having more than 80% missing values
        if len(drop_missing) > 0:
            self.drop_cols_ += drop_missing

        self.drop_cols_ = list(set(self.drop_cols_))

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

        if len(self.drop_cols_) > 0:
            X = X.drop(self.drop_cols_, axis=1)
            print(f'Dropped {len(self.drop_cols_)} columns total in dataset')

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
        self.translated_schema = {}
        for column, dtype in self.schema.items():
            if dtype in ["string","object","category", "str"]:
                self.translated_schema[column] = "object"
            elif dtype in ["text","NLP","nlp"]:
                self.translated_schema[column] = "object"
            elif dtype in ["boolean","bool"]:
                self.translated_schema[column] = "bool"
            elif dtype in [ "np.int8", "int8"]:
                self.translated_schema[column] = "int8"
            elif dtype in ["np.int16","int16"]:
                self.translated_schema[column] = "int16"
            elif dtype in ["int32", "np.int32"]:
                self.translated_schema[column] = "int32"
            elif dtype in ["integer","int", "int64", "np.int64"]:
                self.translated_schema[column] = "int64"
            elif dtype in ["date"]:
                self.translated_schema[column] = "datetime64[ns]"                
            elif dtype in ["float"]:
                self.translated_schema[column] = "float64"
            elif dtype in ["np.float32", "float32"]: 
                self.translated_schema[column] = "float32"
            elif dtype in ["np.float64", "float64"]:
                self.translated_schema[column] = "float64"
            else:
                raise ValueError("Invalid data type: {}".format(dtype))

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
        # Check if the data types of the columns in the dataframe match the data types in the schema
        errors = []
        for column, dtype in self.translated_schema.items():
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
            self.error_df_ = pd.DataFrame()
            print("**No Data Schema Errors**")

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
###################################################################################
from scipy.stats import ks_2samp

def dc_report(train, test, exclude=[], html=False, verbose=0):
    """
    This is a data comparison tool that accepts two pandas dataframes as input and 
    returns a report highlighting any differences between them. You can exclude
    certain columns from this comparison (such as target column) by using
    the "exclude" argument. You can also compare the unique values between two
    dataframes using the compare_uniques argument. 
    
    Parameters
    ----------
    train : pd.DataFrame
        The training dataframe to be compared.
    test : pd.DataFrame
        The testing dataframe to be compared.
    html : False
        You can set it to True to receive an HTML report as well as a file saved in your
        working directory.
    exclude : []
        You can exclude an columns from comparison such as target column which is in train
        but not in test so that the comparison can be 1:1 between the dataframes.
    compare_uniques : []
        If you give column names here, it will compare the number of unique values in those
        columns between the two dataframes. Useful for train test comparisons.
    verbose : 0 or 1
        If 1, Provides a longer report detailing all the columns mentioned in the report output below.
        If 0, Provides only a shorter report with Column Name, DQ Issue Train, DQ Issue Test and Distribution Difference.
    
    Returns
    -------
    report : pd.DataFrame
        A dataframe with the following column names: Column Name, Data Type Train, 
        Data Type Test, Missing Values% Train, Missing Values% Test, Unique Values% Train, 
        Unique Values% Test, Minimum Value Train, Minimum Value Test, 
        Maximum Value Train, Maximum Value Test, DQ Issue Train, DQ Issue Test, 
        Distribution Difference.
        The Distribution Difference column contains comments on any differences between 
        the two dataframes based on the Kolmogorov-Smirnov test statistic for numeric 
        columns with low cardinality, and the percentage of missing values and 
        unique values for all columns.
    
    Raises
    ------
    ValueError
        If the input are not pandas dataframes or if the two dataframes do not have the same columns
        except for the exclude list of columns.
    """
    train = copy.deepcopy(train)
    test = copy.deepcopy(test)

    # Check if the input are pandas dataframes
    if not isinstance(train, pd.DataFrame) or not isinstance(test, pd.DataFrame):
        print("The input must be pandas dataframes. Stopping!")
        return pd.DataFrame()

    ### drop the columns from each 
    if len(exclude) > 0:
        for each in exclude:
            if each in train.columns:
                train = train.drop(each, axis=1)
            else:
                print('Column %s not found in train' %each)
            if each in test.columns:
                test = test.drop(each, axis=1)
            else:
                print('Column %s not found in train' %each)

    # Check if the two dataframes have the same columns
    if not train.columns.equals(test.columns):
        print("The two dataframes dont have the same columns. Use exclude argument to exclude columns from comparison.")
        return pd.DataFrame()
    else:
        print('Analyzing two dataframes for differences. This will take time, please be patient...')
    
    # Use your function dqr = dq_report(df) to generate a data quality report for each dataframe
    dqr_tr = dq_report(data=train, verbose=-1)
    dqr_te = dq_report(data=test,verbose = -1)
    
    # Merge the two reports on the column name
    report = dqr_tr.join(dqr_te, lsuffix="_Train", rsuffix="_Test")
    
    # Initialize an empty list to store the distribution difference results
    dist_diff = []
    
    # Loop through each column in the dataframes
    for col in train.columns:
        # Get the data type of the column in each dataframe
        dtype_train = train[col].dtype
        dtype_test = test[col].dtype
        
        # Get the percentage of missing values in each dataframe
        missing_train = dqr_tr.loc[col, "Missing Values%"]
        missing_test = dqr_te.loc[col, "Missing Values%"]
        
        # Get the percentage of unique values in each dataframe
        unique_train = dqr_tr.loc[col, "Unique Values%"]
        if dqr_tr.loc[col, "Unique Values%"]=='NA':
            count_unique_train = 0
        else:
            count_unique_train = len(train)*(unique_train / 100)
        unique_test = dqr_te.loc[col, "Unique Values%"]
        if dqr_te.loc[col, "Unique Values%"]=='NA':
            count_unique_test = 0
        else:
            count_unique_test = len(test)*(unique_test / 100)
        
        # Initialize an empty string to store the distribution difference comments for the current column
        dist_diff_col = ""
        
        # If the column is numeric and has low cardinality, get the minimum and maximum values from the report dataframe
        if np.issubdtype(dtype_train, np.number) and np.issubdtype(dtype_test, np.number) and count_unique_train < 10 and count_unique_test < 10:
            min_train = report.loc[ col, "Minimum Value_Train"]
            min_test = report.loc[ col, "Minimum Value_Test"]
            max_train = report.loc[ col, "Maximum Value_Train"]
            max_test = report.loc[ col, "Maximum Value_Test"]
            
            # If the column is not missing in both dataframes, compute the Kolmogorov-Smirnov test statistic to measure the distribution difference
            if missing_train < 100 and missing_test < 100:
                ks_stat = ks_2samp(train[col].dropna(), test[col].dropna()).statistic
                
                # If the test statistic is greater than zero, add a comment to indicate that there is a distribution difference
                if ks_stat > 0:
                    dist_diff_col += f"The distributions of {col} are different with a KS test statistic of {ks_stat:.3f}. "
        
        # If the percentage of missing values are different between the two dataframes, add a comment to indicate that there is a missing value difference
        if missing_train != missing_test:
            dist_diff_col += f"The percentage of missing values of {col} are different between train ({missing_train:.2f}%) and test ({missing_test:.2f}%). "

        # If the percentage of unique values are different between the two dataframes, add a comment to indicate that there is a unique value difference
        if unique_train != unique_test:
            if unique_train=='NA' or unique_test == 'NA':
                dist_diff_col += f"The data types of {col} are different between train: {train[col].dtype} and test: {test[col].dtype}. "
            else:
                dist_diff_col += f"The percentage of unique values of {col} are different between train ({unique_train:.2f}%) and test ({unique_test:.2f}%). "
        
        # If the distribution difference comments are empty, set it to None
        if dist_diff_col == "":
            dist_diff_col = None
        
        # Append the distribution difference comments for the current column to the dist_diff list
        dist_diff.append(dist_diff_col)
    
    # Add a new column to the report dataframe with the distribution difference results
    report["Distribution Difference"] = dist_diff
    report = report.reset_index().rename(columns={'index':"Column Name"})

    if verbose:
        if html:
            write_to_html(report, filename="dc_report.html")
        else:
            # Return the report dataframe
            all_rows = report.shape[0]
            ax = report.head(all_rows).style.background_gradient(cmap='Reds').set_properties(**{'font-family': 'Segoe UI'})
            display(ax);
        return report
    else:
        short_report = report[['Column Name','DQ Issue_Train','DQ Issue_Test',"Distribution Difference"]]
        if html:
            write_to_html(short_report, filename="dc_report.html")
        else:
            # Return a shorter version of the dataframe
            all_rows = short_report.shape[0]
            ax = short_report.head(all_rows).style.background_gradient(cmap='Reds').set_properties(**{'font-family': 'Segoe UI'})
            display(ax);
        return short_report
############################################################################################
def classify_columns(df_preds, verbose=0):
    """
    This function does Exploratory data analysis (EDA) to classify columns into types.
    ######################################################################################
    Takes a dataframe containing only predictors to be classified into various types.
    DO NOT SEND IN A TARGET COLUMN since it will try to include that into various columns.
    Returns a data frame containing columns and the class it belongs to such as numeric,
    categorical, date or id column, boolean, nlp, discrete_string and cols to delete...
    ####### Returns a dictionary with 10 kinds of vars like the following: # continuous_vars,int_vars
    # cat_vars,factor_vars, bool_vars,discrete_string_vars,nlp_vars,date_vars,id_vars,cols_delete
    """
    train = copy.deepcopy(df_preds)
    #### If there are 30 chars are more in a discrete_string_var, it is then considered an NLP variable
    max_nlp_char_size = 30
    max_cols_to_print = 30
    #### Cat_Limit defines the max number of categories a column can have to be called a categorical colum
    cat_limit = 35
    float_limit = 15 #### Make this limit low so that float variables below this limit become cat vars ###
    def add(a,b):
        return a+b
    sum_all_cols = dict()
    orig_cols_total = train.shape[1]
    #Types of columns
    cols_delete = []
    cols_delete = [col for col in list(train) if (len(train[col].value_counts()) == 1
                                       ) | (train[col].isnull().sum()/len(train) >= 0.90)]
    inf_cols = EDA_find_remove_columns_with_infinity(train, remove=False, verbose=verbose)
    mixed_cols = [x for x in list(train) if len(train[x].dropna().apply(type).value_counts()) > 1]
    if len(mixed_cols) > 0:
        print('    Removing %s column(s) due to mixed data type detected...' %mixed_cols)
    cols_delete += mixed_cols
    cols_delete += inf_cols
    train = train[left_subtract(list(train),cols_delete)]
    var_df = pd.Series(dict(train.dtypes)).reset_index(drop=False).rename(
                        columns={0:'type_of_column'})
    sum_all_cols['cols_delete'] = cols_delete

    var_df['bool'] = var_df.apply(lambda x: 1 if x['type_of_column'] in ['bool','object']
                        and len(train[x['index']].value_counts()) == 2 else 0, axis=1)
    string_bool_vars = list(var_df[(var_df['bool'] ==1)]['index'])
    sum_all_cols['string_bool_vars'] = string_bool_vars
    var_df['num_bool'] = var_df.apply(lambda x: 1 if x['type_of_column'] in [np.uint8,
                            np.uint16, np.uint32, np.uint64,
                            'int8','int16','int32','int64',
                            'float16','float32','float64'] and len(
                        train[x['index']].value_counts()) == 2 else 0, axis=1)
    num_bool_vars = list(var_df[(var_df['num_bool'] ==1)]['index'])
    sum_all_cols['num_bool_vars'] = num_bool_vars
    ######   This is where we take all Object vars and split them into diff kinds ###
    discrete_or_nlp = var_df.apply(lambda x: 1 if x['type_of_column'] in ['object']  and x[
        'index'] not in string_bool_vars+cols_delete else 0,axis=1)
    ######### This is where we figure out whether a string var is nlp or discrete_string var ###
    var_df['nlp_strings'] = 0
    var_df['discrete_strings'] = 0
    var_df['cat'] = 0
    var_df['id_col'] = 0
    discrete_or_nlp_vars = var_df.loc[discrete_or_nlp==1]['index'].values.tolist()
    copy_discrete_or_nlp_vars = copy.deepcopy(discrete_or_nlp_vars)
    if len(discrete_or_nlp_vars) > 0:
        for col in copy_discrete_or_nlp_vars:
            #### first fill empty or missing vals since it will blowup ###
            ### Remember that fillna only works at the dataframe level!
            train[[col]] = train[[col]].fillna('  ')
            if train[col].map(lambda x: len(x) if type(x)==str else 0).max(
                ) >= 50 and len(train[col].value_counts()
                        ) >= int(0.9*len(train)) and col not in string_bool_vars:
                var_df.loc[var_df['index']==col,'nlp_strings'] = 1
            elif train[col].map(lambda x: len(x) if type(x)==str else 0).mean(
                ) >= max_nlp_char_size and train[col].map(lambda x: len(x) if type(x)==str else 0).max(
                ) < 50 and len(train[col].value_counts()
                        ) <= int(0.9*len(train)) and col not in string_bool_vars:
                var_df.loc[var_df['index']==col,'discrete_strings'] = 1
            elif len(train[col].value_counts()) > cat_limit and len(train[col].value_counts()
                        ) <= int(0.9*len(train)) and col not in string_bool_vars:
                var_df.loc[var_df['index']==col,'discrete_strings'] = 1
            elif len(train[col].value_counts()) > cat_limit and len(train[col].value_counts()
                        ) == len(train) and col not in string_bool_vars:
                var_df.loc[var_df['index']==col,'id_col'] = 1
            else:
                var_df.loc[var_df['index']==col,'cat'] = 1
    nlp_vars = list(var_df[(var_df['nlp_strings'] ==1)]['index'])
    sum_all_cols['nlp_vars'] = nlp_vars
    discrete_string_vars = list(var_df[(var_df['discrete_strings'] ==1) ]['index'])
    sum_all_cols['discrete_string_vars'] = discrete_string_vars
    ###### This happens only if a string column happens to be an ID column #######
    #### DO NOT Add this to ID_VARS yet. It will be done later.. Dont change it easily...
    #### Category DTYPE vars are very special = they can be left as is and not disturbed in Python. ###
    var_df['dcat'] = var_df.apply(lambda x: 1 if str(x['type_of_column'])=='category' else 0,
                            axis=1)
    factor_vars = list(var_df[(var_df['dcat'] ==1)]['index'])
    sum_all_cols['factor_vars'] = factor_vars
    ########################################################################
    date_or_id = var_df.apply(lambda x: 1 if x['type_of_column'] in [np.uint8,
                         np.uint16, np.uint32, np.uint64,
                         'int8','int16',
                        'int32','int64']  and x[
        'index'] not in string_bool_vars+num_bool_vars+discrete_string_vars+nlp_vars else 0,
                                        axis=1)
    ######### This is where we figure out whether a numeric col is date or id variable ###
    var_df['int'] = 0
    var_df['date_time'] = 0
    ### if a particular column is date-time type, now set it as a date time variable ##
    var_df['date_time'] = var_df.apply(lambda x: 1 if x['type_of_column'] in ['<M8[ns]','datetime64[ns]']  and x[
        'index'] not in string_bool_vars+num_bool_vars+discrete_string_vars+nlp_vars else 0,
                                        axis=1)
    ### this is where we save them as date time variables ###
    if len(var_df.loc[date_or_id==1]) != 0:
        for col in var_df.loc[date_or_id==1]['index'].values.tolist():
            if len(train[col].value_counts()) == len(train):
                if train[col].min() < 1900 or train[col].max() > 2050:
                    var_df.loc[var_df['index']==col,'id_col'] = 1
                else:
                    try:
                        pd.to_datetime(train[col],infer_datetime_format=True)
                        var_df.loc[var_df['index']==col,'date_time'] = 1
                    except:
                        var_df.loc[var_df['index']==col,'id_col'] = 1
            else:
                if train[col].min() < 1900 or train[col].max() > 2050:
                    if col not in num_bool_vars:
                        var_df.loc[var_df['index']==col,'int'] = 1
                else:
                    try:
                        pd.to_datetime(train[col],infer_datetime_format=True)
                        var_df.loc[var_df['index']==col,'date_time'] = 1
                    except:
                        if col not in num_bool_vars:
                            var_df.loc[var_df['index']==col,'int'] = 1
    else:
        pass
    int_vars = list(var_df[(var_df['int'] ==1)]['index'])
    date_vars = list(var_df[(var_df['date_time'] == 1)]['index'])
    id_vars = list(var_df[(var_df['id_col'] == 1)]['index'])
    sum_all_cols['int_vars'] = int_vars
    copy_date_vars = copy.deepcopy(date_vars)
    for date_var in copy_date_vars:
        #### This test is to make sure sure date vars are actually date vars
        try:
            pd.to_datetime(train[date_var],infer_datetime_format=True)
        except:
            ##### if not a date var, then just add it to delete it from processing
            cols_delete.append(date_var)
            date_vars.remove(date_var)
    sum_all_cols['date_vars'] = date_vars
    sum_all_cols['id_vars'] = id_vars
    sum_all_cols['cols_delete'] = cols_delete
    ## This is an EXTREMELY complicated logic for cat vars. Don't change it unless you test it many times!
    var_df['numeric'] = 0
    float_or_cat = var_df.apply(lambda x: 1 if x['type_of_column'] in ['float16',
                            'float32','float64'] else 0,
                                        axis=1)
    #######  We need to make sure there are no categorical vars in float #######
    if len(var_df.loc[float_or_cat == 1]) > 0:
        for col in var_df.loc[float_or_cat == 1]['index'].values.tolist():
            if len(train[col].value_counts()) > 2 and len(train[col].value_counts()
                ) <= float_limit and len(train[col].value_counts()) <= len(train):
                var_df.loc[var_df['index']==col,'cat'] = 1
            else:
                if col not in (num_bool_vars + factor_vars):
                    var_df.loc[var_df['index']==col,'numeric'] = 1
    cat_vars = list(var_df[(var_df['cat'] ==1)]['index'])
    continuous_vars = list(var_df[(var_df['numeric'] ==1)]['index'])

    ########  V E R Y    I M P O R T A N T   ###################################################
    cat_vars_copy = copy.deepcopy(factor_vars)
    for cat in cat_vars_copy:
        if df_preds[cat].dtype==float:
            continuous_vars.append(cat)
            factor_vars.remove(cat)
            var_df.loc[var_df['index']==cat,'dcat'] = 0
            var_df.loc[var_df['index']==cat,'numeric'] = 1
        elif len(df_preds[cat].value_counts()) == df_preds.shape[0]:
            id_vars.append(cat)
            factor_vars.remove(cat)
            var_df.loc[var_df['index']==cat,'dcat'] = 0
            var_df.loc[var_df['index']==cat,'id_col'] = 1

    sum_all_cols['factor_vars'] = factor_vars
    ##### There are a couple of extra tests you need to do to remove abberations in cat_vars ###
    cat_vars_copy = copy.deepcopy(cat_vars)
    for cat in cat_vars_copy:
        if df_preds[cat].dtype==float:
            continuous_vars.append(cat)
            cat_vars.remove(cat)
            var_df.loc[var_df['index']==cat,'cat'] = 0
            var_df.loc[var_df['index']==cat,'numeric'] = 1
        elif len(df_preds[cat].value_counts()) == df_preds.shape[0]:
            id_vars.append(cat)
            cat_vars.remove(cat)
            var_df.loc[var_df['index']==cat,'cat'] = 0
            var_df.loc[var_df['index']==cat,'id_col'] = 1
    sum_all_cols['cat_vars'] = cat_vars
    sum_all_cols['continuous_vars'] = continuous_vars
    sum_all_cols['id_vars'] = id_vars
    ###### This is where you consoldate the numbers ###########
    var_dict_sum = dict(zip(var_df.values[:,0], var_df.values[:,2:].sum(1)))
    for col, sumval in var_dict_sum.items():
        if sumval == 0:
            print('%s of type=%s is not classified' %(col,train[col].dtype))
        elif sumval > 1:
            print('%s of type=%s is classified into more then one type' %(col,train[col].dtype))
        else:
            pass
    ##### If there are more than 1000 unique values, then add it to NLP vars ###
    copy_discretes = copy.deepcopy(discrete_string_vars)
    for each_discrete in copy_discretes:
        if train[each_discrete].nunique() >= 1000:
            nlp_vars.append(each_discrete)
            discrete_string_vars.remove(each_discrete)
        elif train[each_discrete].nunique() > 100 and train[each_discrete].nunique() < 1000:
            pass
        else:
            ### If it is less than 100 unique values, then make it categorical var
            cat_vars.append(each_discrete)
            discrete_string_vars.remove(each_discrete)
    sum_all_cols['discrete_string_vars'] =  discrete_string_vars
    sum_all_cols['cat_vars'] = cat_vars
    sum_all_cols['nlp_vars'] = nlp_vars
    ###############  This is where you print all the types of variables ##############
    ####### Returns 8 vars in the following order: continuous_vars,int_vars,cat_vars,
    ###  string_bool_vars,discrete_string_vars,nlp_vars,date_or_id_vars,cols_delete
    if verbose == 1:
        print("    Number of Numeric Columns = ", len(continuous_vars))
        print("    Number of Integer-Categorical Columns = ", len(int_vars))
        print("    Number of String-Categorical Columns = ", len(cat_vars))
        print("    Number of Factor-Categorical Columns = ", len(factor_vars))
        print("    Number of String-Boolean Columns = ", len(string_bool_vars))
        print("    Number of Numeric-Boolean Columns = ", len(num_bool_vars))
        print("    Number of Discrete String Columns = ", len(discrete_string_vars))
        print("    Number of NLP String Columns = ", len(nlp_vars))
        print("    Number of Date Time Columns = ", len(date_vars))
        print("    Number of ID Columns = ", len(id_vars))
        print("    Number of Columns to Delete = ", len(cols_delete))
    if verbose >= 2:
        print('  Printing upto %d columns (max) in each category:' %max_cols_to_print)
        print("    Numeric Columns : %s" %continuous_vars[:max_cols_to_print])
        print("    Integer-Categorical Columns: %s" %int_vars[:max_cols_to_print])
        print("    String-Categorical Columns: %s" %cat_vars[:max_cols_to_print])
        print("    Factor-Categorical Columns: %s" %factor_vars[:max_cols_to_print])
        print("    String-Boolean Columns: %s" %string_bool_vars[:max_cols_to_print])
        print("    Numeric-Boolean Columns: %s" %num_bool_vars[:max_cols_to_print])
        print("    Discrete String Columns: %s" %discrete_string_vars[:max_cols_to_print])
        print("    NLP text Columns: %s" %nlp_vars[:max_cols_to_print])
        print("    Date Time Columns: %s" %date_vars[:max_cols_to_print])
        print("    ID Columns: %s" %id_vars[:max_cols_to_print])
        print("    Columns that will not be considered in modeling: %s" %cols_delete[:max_cols_to_print])
    ##### now collect all the column types and column names into a single dictionary to return!

    len_sum_all_cols = reduce(add,[len(v) for v in sum_all_cols.values()])
    if len_sum_all_cols == orig_cols_total:
        print('    All variables classified into correct types.' )
        #print('        This does not include the Target column(s)')
    else:
        print('No of columns classified %d does not match %d total cols. Continuing...' %(
                   len_sum_all_cols, orig_cols_total))
        ls = sum_all_cols.values()
        flat_list = [item for sublist in ls for item in sublist]
        if len(left_subtract(list(train),flat_list)) == 0:
            print(' Missing columns = None')
        else:
            print(' Missing columns = %s' %left_subtract(list(train),flat_list))
    return sum_all_cols
#################################################################################
from functools import reduce
import copy
import time
def left_subtract(l1,l2):
    lst = []
    for i in l1:
        if i not in l2:
            lst.append(i)
    return lst
#################################################################################
import copy
def EDA_find_remove_columns_with_infinity(df, remove=False, verbose=0):
    """
    This function finds all columns in a dataframe that have inifinite values (np.inf or -np.inf)
    It returns a list of column names. If the list is empty, it means no columns were found.
    If remove flag is set, then it returns a smaller dataframe with inf columns removed.
    """
    nums = df.select_dtypes(include='number').columns.tolist()
    dfx = df[nums]
    sum_rows = np.isinf(dfx).values.sum()
    add_cols =  list(dfx.columns.to_series()[np.isinf(dfx).any()])
    if sum_rows > 0:
        if verbose > 0:
            print('    there are %d rows and %d columns with infinity in them...' %(sum_rows,len(add_cols)))
        if remove:
            ### here you need to use df since the whole dataset is involved ###
            nocols = [x for x in df.columns if x not in add_cols]
            if verbose > 0:
                print("    Shape of dataset before %s and after %s removing columns with infinity" %(df.shape,(df[nocols].shape,)))
            return df[nocols]
        else:
            ## this will be a list of columns with infinity ####
            return add_cols
    else:
        ## this will be an empty list if there are no columns with infinity
        return add_cols
####################################################################################
module_type = 'Running' if  __name__ == "__main__" else 'Imported'
version_number =  '1.29'
#print(f"""{module_type} pandas_dq ({version_number}). Always upgrade to get latest features.""")
#################################################################################

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
# Define a function to print data cleaning suggestions
def find_dq(df, target=None, verbose=0):
    if not verbose:
        print("Set verbose to 1 to see more details on each of these data quality issues.")
    # Detect missing values and suggests to impute them with mean, median, mode, or a constant value123
    missing_values = df.isnull().sum()
    missing_cols = missing_values[missing_values > 0].index.tolist()
    if len(missing_cols) > 0:
        print(f"There are {len(missing_cols)} columns with missing values in the dataset. You may want to impute them with mean, median, mode, or a constant value such as 123.")
        if verbose:
            for col in missing_cols:
                print(f"{col} has {missing_values[col]} missing values.")
    else:
        print("There are no columns with missing values in the dataset")

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
        if len(rare_cat_cols) > 0:
            print(f"There are {len(cat_cols)} categorical columns with rare categories in the dataset. You may want to group rare categories into a single category or drop the categories.")
            for rare_col in rare_cat_cols:
                if verbose:
                    print(f"{col} has {len(rare_values)} rare categories: {rare_values}")
    else:
        print(f"There are no categorical columns with rare categories (< {100*rare_threshold:.0f} percent) in this dataset")

    # Find infinite values and suggests to replace them with NaN or a large value123
    inf_values = df.replace([np.inf, -np.inf], np.nan).isnull().sum() - missing_values
    inf_cols = inf_values[inf_values > 0].index.tolist()
    if len(inf_cols) > 0:
        print(f"There are {len(inf_cols)} columns with infinite values in the dataset. You may want to replace them with NaN or a large value.")
        if verbose:
            for col in inf_cols:
                print(f"{col} has {inf_values[col]} infinite values.")
    else:
        print("There are no numeric columns with infinite values in this dataset")

    # Detect mixed data types and suggests to convert them to a single type or split them into multiple columns123
    mixed_types = df.applymap(type).nunique() # Get the number of unique types in each column
    mixed_cols = mixed_types[mixed_types > 1].index.tolist() # Get the columns with more than one type
    if len(mixed_cols) > 0:
        print(f"There are {len(mixed_cols)} columns with mixed data types in the dataset. You may want to convert them to a single type or split them into multiple columns.")
        if verbose:
            for col in mixed_cols:
                if verbose:
                    print(f"{col} has {mixed_types[col]} different types: {df[col].apply(type).unique()}")
    else:
        print("There are no columns with mixed (more than one) dataypes in this dataset")

                
    # Detect duplicate rows and columns
    dup_rows = df.duplicated().sum()
    dup_cols = df.T.duplicated().sum()
    if dup_rows > 0:
        print(f"There are {len(dup_rows)} duplicate rows in the dataset. You may want to drop them or keep only one copy.")
        if verbose:
            print(f"    Duplicate rows: {dup_rows}")
    else:
        print("There are no duplicate rows in this dataset")
    if dup_cols > 0:
        print(f"There are {len(dup_cols)} duplicate columns in the dataset. You may want to drop them or keep only one copy.")
        if verbose:
            print(f"    Duplicate columns: {dup_cols}")
    else:
        print("There are no duplicate columns in this datatset")

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
                    print(f"There are {len(num_cols)} numerical columns, some with outliers. Remove them or use robust statistics.")
                    first_time =False
                ### check if there are outlier columns and print them ##
                if verbose:
                    print(f"    {col} has {len(outliers)} outliers.")
                    print(f"        Here are the values: {outliers.values}")
        if len(outlier_cols) < 1:
            print("There are no numeric columns with outliers in this dataset")
        else:
            if not first_time and not verbose:
                print(f"    {len(outlier_cols)} columns with outliers.")
                
    # Detect high cardinality features
    cardinality_threshold = 100 # Define a threshold for high cardinality
    cardinality = df.nunique() # Get the number of unique values in each column
    high_card_cols = cardinality[cardinality > cardinality_threshold].index.tolist() # Get the columns with high cardinality
    if len(high_card_cols) > 0:
        print(f"There are {len(high_card_cols)} columns with high cardinality (>{cardinality_threshold}) in the dataset. You may want to reduce them using encoding techniques or feature selection methods.")
        for col in high_card_cols:
            if verbose:
                print(f"    {col} has {cardinality[col]} unique values.")

    # Detect highly correlated features
    correlation_threshold = 0.8 # Define a threshold for high correlation
    correlation_matrix = df.corr().abs() # Get the absolute correlation matrix of numerical columns
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)) # Get the upper triangle of the matrix
    high_corr_cols = [column for column in upper_triangle.columns if any(upper_triangle[column] > correlation_threshold)] # Get the columns with high correlation
    if len(high_corr_cols) > 0:
        print(f"There are {len(high_corr_cols)} columns with higher than {correlation_threshold} correlation in the dataset. You may want to drop one of them or use dimensionality reduction techniques.")
        for col in high_corr_cols:
            if verbose:
                print(f"    {col} has a high correlation with {upper_triangle[col][upper_triangle[col] > correlation_threshold].index.tolist()}")
    else:
        print('There are no highly correlated columns in the dataset.')

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
                    print(f"The classes are imbalanced in the {each_target_col} variable. You may want to use resampling techniques or class weights to address this issue.")
                    # Print the class frequencies
                    print(f"Class frequencies:\n{value_counts}")
            
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
            print(f"There are {len(leakage_cols)} columns with feature leakage in the dataset. You may want to avoid using features that are not available at prediction time.")
            for col in leakage_cols:
                if verbose:
                    print(f"    {col} has a correlation higher than {leakage_threshold} with {target_col}")
        else:
            print('There are no target leakage columns in the dataset')
    else:
        print('There is no target given. Hence no target leakage columns detected in the dataset')

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
        num_cols = X.select_dtypes(include=["int", "float"]).columns.tolist()
        
        # Loop through each numerical column
        for col in num_cols:
            # Check if the column has an upper bound calculated in the fit method
            if col in self.upper_bounds_:
                # Cap the outliers using the upper bound
                X[col] = np.where(X[col] > self.upper_bounds_[col], self.upper_bounds_[col], X[col])
        
        # Return the DataFrame with capped outliers
        return X
    
    # Define a function to impute the missing values in categorical and numerical columns using the constant values
    def impute_missing(self, X):
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
        X[cat_cols] = X[cat_cols].fillna(self.cat_fill_value)
        
        # Impute the missing values in numerical columns with the num_fill_value
        X[num_cols] = X[num_cols].fillna(self.num_fill_value)
        
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

    # Define a function to detect duplicate rows and columns and keep only one copy
    def drop_duplicates(self, X):
        # Check if X is a pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            # Convert X to a pandas DataFrame
            X = pd.DataFrame(X)
        
        # Drop duplicate rows
        dup_rows = X.duplicated().sum()
        X = X.drop_duplicates()
        if dup_rows > 0:
            print(f'Dropping {dup_rows} rows')
        
        # Drop duplicate columns
        dup_cols = X.T.duplicated().sum()
        X = X.T.drop_duplicates().T
        if dup_cols > 0:
            print(f'Dropping {dup_cols} cols')
        
        # Return the DataFrame with no duplicates
        return X
    
    # Define a function to detect skewed distributions and apply a proper transformation to the column
    def transform_skewed(self, X):
        # Check if X is a pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            # Convert X to a pandas DataFrame
            X = pd.DataFrame(X)
        
        # Get the numerical columns
        num_cols = X.select_dtypes(include=["int", "float"]).columns.tolist()
        
        # Define a threshold for skewness
        skew_threshold = 0.5
        
        # Loop through each numerical column
        for col in num_cols:
            # Find if a column transformer exists for this column
            if col in self.col_transformers_:
                # Cap the outliers using the upper bound
                # Check if the skewness is above the threshold
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
        for col in num_cols:
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
        skew_threshold = 0.5
        
        # Loop through each numerical column
        for col in num_cols:
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
            
        ### drop mixed data type columns from further processing ##
        if len(self.mixed_type_cols_) > 0:
            X = X.drop(self.mixed_type_cols_, axis=1)
            
        ### drop highly correlated columns from further processing ##
        if len(self.drop_corr_cols_) > 0:
            if len(left_subtract(self.drop_corr_cols_,self.mixed_type_cols_)) > 0:
                extra_cols = left_subtract(self.drop_corr_cols_,self.mixed_type_cols_)
            elif len(left_subtract(self.mixed_type_cols_,drop_corr_cols_)) > 0:
                extra_cols = left_subtract(self.mixed_type_cols_, self.drop_corr_cols_)
            if len(extra_cols) > 0:
                X = X.drop(extra_cols, axis=1)
            
        # find duplicate columns and rows
        X = self.drop_duplicates(X)
        
        # Call the cap_outliers function on X and assign it to a new variable 
        capped_X = self.cap_outliers(X)
        
        # Call the replace_infinite function on capped_X and assign it to a new variable 
        infinite_X = self.replace_infinite(capped_X)
        
        # Call the group_rare_categories function on infinite_X and assign it to a new variable 
        rare_X = self.group_rare_categories(infinite_X)
        
        # Call the power transformer function on rare_X and assign it to a new variable 
        power_X = self.transform_skewed(rare_X)
        
        # Call the impute_missing function on power_X and assign it to a new variable 
        imputed_X = self.impute_missing(power_X)
        
        # Return the transformed DataFrame
        return imputed_X

############################################################################################
module_type = 'Running' if  __name__ == "__main__" else 'Imported'
version_number =  '1.1'
print(f"""{module_type} pandas_dq ({version_number}). Use fit and transform using:
from pandas_dq import find_dq, Fix_DQ
fdq = Fix_DQ(quantile=0.75, cat_fill_value="missing", num_fill_value=9999, 
                 rare_threshold=0.05, correlation_threshold=0.8)
""")
#################################################################################

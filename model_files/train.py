import numpy as np
import pandas as pd
import datetime

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as imbPipeline
import joblib

from model_files import config
dateCols = config.DATE_COLS
derivedCols = config.DERIVED_COLS
constantImputation = config.CONSTANT_IMPUTATION


# features_object = './model_files/feature_engineering.joblib'
# features_transform_pipe = joblib.load(features_object)

# process datetime columns
def preprocess_datetime(df, dateCols):
    '''
    Function to convert the variables with temporal information in raw data to datetime  datatype
    Argument:
        df: The raw data read as a dataframe
        dateCols: The list of columns which has to be converted to datetime datatypes
    Returns:
        df: A dataframe with variable containing temporal information as datetime datatype
    '''
    df[dateCols] = df[dateCols].applymap(lambda x : x * 10).applymap(lambda x : str(x)).applymap(lambda x : datetime.datetime.strptime(x, "%Y%W%w" ))
    return df


def preprocess_string(df):
    '''
    Function to standardize the object/string variables (lower case and remove leading 
            and trailing blank spaces for consistency) and convert all string columns to CAtegorical data type 
            to reduce memory occupied by dataframe making the training process more efficient
    Argument:
        df: The raw data read as a dataframe
    Returns:
        df: A dataframe with variable containing temporal information as datetime datatype
    '''
    
    cat_cols = df.select_dtypes(include = 'object').columns.tolist()
    df[cat_cols] = df[cat_cols].applymap(lambda x: x.upper().strip() if not pd.isna(x) else x)
    df[cat_cols] = df[cat_cols].apply(lambda x : x.astype('category'))
    return df

class CustomAttrAdder(BaseEstimator, TransformerMixin):
    def __init__(self, product_age_indays=True, ): # no *args or **kargs
        self.product_age_indays = product_age_indays
    
    def fit(self, X, y=None):
        return self  # nothing else to do
    
    @staticmethod
    def getIndex(df, dateCols):
        datecols_index =  {col : index for index, col in enumerate(df) if col in dateCols}
                
        return datecols_index['mnfture_wk'], datecols_index['contract_st'], datecols_index['contract_end'] , datecols_index['contact_wk'] 

    def transform(self, data, X):
        mnfture_wk_ix, contract_st_ix, contract_end_ix, contact_wk_ix  = self.getIndex(data, dateCols)
        contract_age_indays = np.fromiter((d.days for d in (X[:,contact_wk_ix ] - X[:, contract_st_ix])), dtype=int, count=len((X[:,contact_wk_ix ] - X[:, contract_st_ix])))
        
        contract_st_delay_indays = np.fromiter((d.days for d in (X[:,contract_st_ix] - X[:, mnfture_wk_ix])), dtype=int, count=len((X[:,contract_st_ix] - X[:, mnfture_wk_ix])))
        contract_remaining_indays = np.fromiter((d.days for d in (X[:,contract_end_ix] - X[:, contact_wk_ix])), dtype=int, count=len((X[:,contract_end_ix] - X[:, contact_wk_ix])))
        contract_length_indays = np.fromiter((d.days for d in (X[:,contract_end_ix ] - X[:, contract_st_ix])), dtype=int, count=len((X[:,contract_end_ix ] - X[:, contract_st_ix])))
        
        if self.product_age_indays:
            product_age_indays = np.fromiter((d.days for d in (X[:, contact_wk_ix] - X[:, mnfture_wk_ix])), dtype=int, count=len((X[:, contact_wk_ix] - X[:, mnfture_wk_ix])))
            return pd.DataFrame(np.c_[ product_age_indays, contract_age_indays, contract_st_delay_indays, contract_remaining_indays, contract_length_indays], columns = derivedCols)
        
        return pd.DataFrame(np.c_[contract_age_indays, contract_st_delay_indays, contract_remaining_indays, contract_length_indays], columns = derivedCols)


def preprocess_data(data, dateCols):
    
    preprocess_datetime(data, dateCols)
    # preprocess string
    
    preprocess_string(data)

    attr_adder = CustomAttrAdder(product_age_indays=True)
    data_extra_attrs = attr_adder.transform(data, data.values)
    data.drop(columns= dateCols, inplace = True)
    data=  pd.concat([data , data_extra_attrs], axis = 1)
    
    return data




def num_pipeline_transformer(data):
    '''
    Function to process numerical transformations
    Argument:
        data: original dataframe 
    Returns:
        num_attrs: numerical variable list
        num_pipeline: numerical pipeline object
        
    '''
    numerical_columns = [col for col in list(data.iloc[:0,:].select_dtypes(exclude = 'category').columns) if col.lower().strip() not in ['contact_manager_flg', 'repeat_ct']]
   
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
        ])
    return numerical_columns, num_pipeline


def cat_pipeline_transformer(data):
    '''
    Function to process categorical variables transformations
    Argument:
        data: original dataframe 
    Returns:
        cat_columns: all cataegorical variables list which require mode imputation       
        categorical_transformer: Categorical variables pipeline object for categorical variables
        which requires mode imputation
        
        constant_imp_columns: all cataegorical variables list which require contsnat imputation       
        imputer_categoric_contsant: Categorical variables pipeline object for categorical variables
        which requires constant imputation
        
    '''
    #all categorical columns
    cat_columns  = list(data.iloc[:0,:].select_dtypes(include = 'category').columns)
    # all categorical columns which require constant imputation
    constant_imp_columns = [col for col in cat_columns if col  in constantImputation]
    # all categorical columns which do not require constant imputation ,require mode imputation
    cat_columns = list(set(cat_columns)^constantImputation)    

    
    imputer_categoric_contsant = Pipeline(
    steps=[('const_imputer',SimpleImputer(strategy='constant', fill_value='None')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse = False))])
    
    
    
    categorical_transformer = Pipeline(steps = [
        ('cat_imp',SimpleImputer(strategy = 'most_frequent') ),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse = False))])
    
    return cat_columns, categorical_transformer ,constant_imp_columns,  imputer_categoric_contsant


def pipeline_transformer(data):
    '''
    Complete transformation pipeline for both
    nuerical and categorical data.
    
    Argument:
        data: original dataframe 
    Returns:
        prepared_data: transformed data, ready to use
    '''
   

    num_attrs, num_pipeline = num_pipeline_transformer(data)
    
    cat_columns, categorical_transformer ,constant_imp_columns,  imputer_categoric_contsant = cat_pipeline_transformer(data)
    
        
    col_transformer = ColumnTransformer([
    ("num", num_pipeline, list(num_attrs) ),
    ("imp_cat_constant", imputer_categoric_contsant, constant_imp_columns),
    ("cat", categorical_transformer, cat_columns )        
    ],remainder = 'passthrough')
    
    prepared_data = features_transform_pipe.fit(data)
    return prepared_data


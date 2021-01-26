from model_files import config
import numpy as np
import pandas as pd
import datetime

from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import logging

logging.basicConfig(filename = "repeat_customers_visit_process.log", level = logging.WARNING,
                    format = '%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

dateCols = config.DATE_COLS



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
    """
        Class to create derived columns (custom attributes) from the incoming data
        which is fed as an input feature to the classifier model
        Inherits the BaseEstimator and Transformixin class and define the transform method 
        for our custom transformer
    """
    def __init__(self, product_age_indays=True, ): # no *args or **kargs
        self.product_age_indays = product_age_indays
    
    def fit(self, X, y=None):
        return self  # nothing else to do
    
    @staticmethod
    def getIndex(df, dateCols):
        """
        Function to get index of columns used to create derived colums
        Arguements: 
            df : the incoming data
            dateCols: the list of columns that are used in deriving custom columns
        
        Returns:
                the index of all columns in df used in creating custom columns
        """
        datecols_index =  {col : index for index, col in enumerate(df) if col in dateCols}
                
        return datecols_index['mnfture_wk'], datecols_index['contract_st'], datecols_index['contract_end'] , datecols_index['contact_wk'] 

    def transform(self, data, X):
        """
        Function to create derived columns

        Arguement: 
            data: incoming data as a dataframe
            X: the incoming data as numpy array

        Returns: A dataframe object containing the derived columns
        """
        mnfture_wk_ix, contract_st_ix, contract_end_ix, contact_wk_ix  = self.getIndex(data, dateCols)
        contract_age_indays = np.fromiter((d.days for d in (X[:,contact_wk_ix ] - X[:, contract_st_ix])), dtype=int, count=len((X[:,contact_wk_ix ] - X[:, contract_st_ix])))
        
        contract_st_delay_indays = np.fromiter((d.days for d in (X[:,contract_st_ix] - X[:, mnfture_wk_ix])), dtype=int, count=len((X[:,contract_st_ix] - X[:, mnfture_wk_ix])))
        contract_remaining_indays = np.fromiter((d.days for d in (X[:,contract_end_ix] - X[:, contact_wk_ix])), dtype=int, count=len((X[:,contract_end_ix] - X[:, contact_wk_ix])))
        contract_length_indays = np.fromiter((d.days for d in (X[:,contract_end_ix ] - X[:, contract_st_ix])), dtype=int, count=len((X[:,contract_end_ix ] - X[:, contract_st_ix])))
        
        if self.product_age_indays:
            product_age_indays = np.fromiter((d.days for d in (X[:, contact_wk_ix] - X[:, mnfture_wk_ix])), dtype=int, count=len((X[:, contact_wk_ix] - X[:, mnfture_wk_ix])))
            return pd.DataFrame(np.c_[ product_age_indays, contract_age_indays, contract_st_delay_indays, contract_remaining_indays, contract_length_indays], columns = config.DERIVED_COLS)
        
        return pd.DataFrame(np.c_[contract_age_indays, contract_st_delay_indays, contract_remaining_indays, contract_length_indays], columns = config.DERIVED_COLS)


def preprocess_data(data, dateCols):
    """
        Function to apply all transformations to the dataframe i.e. convert date related columns
        to datetime, standardize strings, convert these columns from object type to category for 
        faster processing and reduce memory usage and create derived columns from original elements
        in the json request

        Arguement: 
            data : incoming data as dataframe
            dateCols : a list of date columns

        Returns: 
            data: a dataframe object with all transofmations applied
    """
    
    preprocess_datetime(data, dateCols)
    logging.warning("The datetime columns has been preprocessed fine")

    
    preprocess_string(data)
    logging.warning("The string column has been preprocessed")

    attr_adder = CustomAttrAdder(product_age_indays=True)
    data_extra_attrs = attr_adder.transform(data, data.values)
    data.drop(columns= dateCols, inplace = True)
    data=  pd.concat([data , data_extra_attrs], axis = 1)
    logging.warning("The custom attributed has been created")
    
    return data


def predict_repeat_contact(inputs, model, features_transform_pipe):
    """
        Function to return the predictions and scoring  from a trained classifier 
         after all feature engineering steps are applied from the saved pipeline 
        of an incoming record

        Arguement:
            inputs : the incoming input data from the API
            model : trained model object
            features_transform_pipe : saved feature transformations pipeline

    """
    hashmap = config.HASH_MAP
    if type(inputs)== dict:
        df = pd.DataFrame(inputs)
    else:
        df = inputs  
   
    preproc_df = preprocess_data(df, dateCols)
    transformed_df = features_transform_pipe.transform(preproc_df)
    logging.warning("The features transformations has been applied")
    incoming_input_pred = model.predict(transformed_df)
    incoming_input_score = model.predict_proba(transformed_df)[:,1]
    
    incoming_input_label = hashmap[incoming_input_pred[0]]
    
    return {"Predicted Class": incoming_input_label,
            "Score": f"{incoming_input_score[0] *100:.2f}" + "%" }

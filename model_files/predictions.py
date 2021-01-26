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
derivedCols = config.DERIVED_COLS
constantImputation = config.CONSTANT_IMPUTATION

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

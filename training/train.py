import numpy as np
import pandas as pd
import datetime

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt

import joblib
from imblearn import over_sampling, under_sampling
from sklearn import metrics
import logging
import config

dateCols = config.DATE_COLS
derivedCols = config.DERIVED_COLS
#constantImputation = config.CONSTANT_IMPUTATION

# logging  
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('train.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)



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


def binarizeTarget(df, targetvar):
    '''
    Function to convert the multiclass target variable to binary variable
    Argument:
        df: The raw data read as a dataframe
        targetvar: target variable for which the classification model is built
    Returns:
        df: A dataframe containing binarized target
    '''
    
    df[targetvar] = df[targetvar].apply(lambda x: 1 if x > 0 else 0)
    return df


def preprocess_data(rawData, dateCols):
   
    '''
    Function to read the raw data into a Pandas dataframe and apply transformations 
    Argument:
        rawdata: raw data file name 
        dateCols: The list of columns which has to be converted to datetime datatypes
    Returns:
        data: the dataframe with appropriate data types making it smaller and faster to run from time and space complexity
        standpoint
       
    '''

    # get list of column names by just reading one row of data- a constant time operation
    columns = list(pd.read_csv(rawData, nrows=1 ))
    
    #global columnsNotToRead
    data = pd.read_csv(rawData, usecols = [ col for col in columns if col not in config.NOT_TO_READ ])

    preprocess_datetime(data, dateCols)
    # preprocess string
    preprocess_string(data)
    
    # binarize the target variable
    
    binarizeTarget(data, config.TARGETVAR)

    attr_adder = CustomAttrAdder(product_age_indays=True)
    data_extra_attrs = attr_adder.transform(data, data.values)
    data.drop(columns= dateCols, inplace = True)
    data=  pd.concat([data , data_extra_attrs], axis = 1) 
       
    return data


def split_data(df):
    """
    Function to split data into train and test
    Arguement: 
        df : Dataframe after being cleaned with derived variables
    Returns:
        X_train: Training input set
        X_test: Test Set
        y_train: Training labels
        y_test: test_labels
    
    """
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns = ['repeat_ct']), df['repeat_ct'], test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test 


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
    constant_imp_columns = [col for col in cat_columns if col  in config.CONSTANT_IMPUTATION]
    # all categorical columns which do not require constant imputation ,require mode imputation
    cat_columns = list(set(cat_columns)^config.CONSTANT_IMPUTATION)    

    
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
    
    prepared_data = col_transformer.fit(data)
    return prepared_data

def resampling():
    """
    Function to do oversampling and undersampling
    Retuns:
        sampling_pipeline a samapling piepleine with steps as oversampling and undersampling
    """
    oversampling = over_sampling.SMOTE(random_state=42)
    undersampling = under_sampling.RandomUnderSampler(random_state=42)


    sampling_pipeline = imbPipeline([
    ('oversample', oversampling),
    ('undersample', undersampling)
    ])
    return sampling_pipeline


def train_model():
    data = preprocess_data(config.RAWDATA, dateCols)
    logger.debug("The raw data has been standardized, the dataframe has become memory and time efficient and the computed columns are derived ")
    X_train, X_test, y_train, y_test = split_data(data)
    logger.debug("Train test split is done")
    fitted_transformations = pipeline_transformer(X_train)
    logger.debug("The pipelines on training set is created")
    
    X_train_normalized = fitted_transformations.transform(X_train)
    logger.debug("The trainset has  been  transformed by fitted transformer pipeline")
 
    sampling_pipeline = resampling()
    
    X_train_balanced, y_train_balanced = sampling_pipeline.fit_resample(X_train_normalized, y_train)
    logger.debug("Over and Undersampling is completed")
    
    model.fit(X_train_balanced, y_train_balanced)
    logger.debug("Model has been trained")
    return  fitted_transformations, X_test, y_test

def evaluate_model():
    """
    Function which applies fitted transformation from training set on test data and 
         evaluates the fitted model on test set 

        Returns: Stores the classification report , roc auc curve and the test and 
        actual predictions to an output file  
    """
    fitted_transformations, X_test, y_test = train_model()
    X_test_normalized = fitted_transformations.transform(X_test)
    logger.debug("The test set has  been  transformed by fitted transformer pipeline")
 

    pred = model.predict(X_test_normalized)
    score = model.predict_proba(X_test_normalized)[:, 1]
    logger.debug("The test set has been predicted and scored")
 
    
    classification_rep = metrics.classification_report(y_test, pred)
    
    with open("./class_report.txt", "w+") as f:
        f.write(classification_rep)
    
    auc = metrics.roc_auc_score(y_test, score)
    fpr, tpr, _ = metrics.roc_curve(y_test, score)

    plt.plot(fpr, tpr, label='AUC = %.3f' % auc)
    plt.plot([0, 1], [0, 1],'r--', label='Random')
    plt.legend(loc="lower right")
    
    plt.savefig("./roc_auc_curve.png")
    try:
        output = pd.concat([y_test, pd.DataFrame(pred, columns = ['Predicted_repeated_contact']) ], axis = 1)
    except Exception:
        logger.exception("Concatenating numpy array with pandas dataframe") 

    output.to_csv("./output_predictions.csv", index = False)

    joblib.dump(model, './model.joblib')
    joblib.dump(fitted_transformations, "./feature_engineering.joblib")
    return output

    
if __name__ == "__main__":
    model = LogisticRegression(C=0.1, max_iter=500)
    evaluate_model()
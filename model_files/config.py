

"""
This module contains the list of few variables defined in the data
which should be controlled outside of the application and main code
to improve modularity of the code base
"""

DATE_COLS = ['mnfture_wk', 'contract_st', 'contract_end', 'contact_wk']
DERIVED_COLS = ['product_age_indays', 'contract_age_indays', 'contract_st_delay_indays', 'contract_remaining_indays', 'contract_length_indays' ]
CONSTANT_IMPUTATION = set(['parts_sent'])
FEATURE_TRANSFORMATION_OBJECT_PATH = "feature_engineering.joblib"
MODEL_PATH = "repeat_ct_model.joblib"
HASH_MAP = {0: 'No Contact',
           1: 'Contact'}

NOT_TO_READ = ["asst_id", "repeat_parts_sent"]
# features_transform_pipe = joblib.load(features_object)
# model = joblib.load(model_object)


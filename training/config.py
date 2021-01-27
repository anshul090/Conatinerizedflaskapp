

"""
        Configuration file to be used in training 

This module  contains the list of few variables defined in the data
which should be controlled outside of the application and main code
to improve modularity of the code base
"""
RAWDATA = "Dell_SDS_MLE_takehome_dataset.csv"
TARGETVAR = 'repeat_ct'
NOT_TO_READ = ["asst_id", "repeat_parts_sent"]
DATE_COLS = ['mnfture_wk', 'contract_st', 'contract_end', 'contact_wk']
DERIVED_COLS = ['product_age_indays', 'contract_age_indays', 'contract_st_delay_indays', 'contract_remaining_indays', 'contract_length_indays' ]
CONSTANT_IMPUTATION = set(['parts_sent'])



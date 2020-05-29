import pandas as pd 
import numpy as np 
import pickle 

from src.clean_new_data import clean_new_data

def load_clean_into_pandas(request_info):
    '''
    inputs
    ------
    request_info: json formatted info that will be formatted into 
    '''
    df = clean_new_data(df)

    return df


def predict(request_info):
    '''
    inputs
    ------
    request_info: json formatted info that will be used to make new predictions
    '''

    # load request_info into a pandas dataframe and clean it using 
    df = pd.read_json(request_info)
    df_cleaned = clean_new_data(request_info)

    X_test = np.array(df_cleaned)

    with open('models/grad_boost_model.p') as mod:
        model = pickle.load(mod)

    probs = model.predict_proba(X_test)[:, 1]

    df['fraud_prob'] = probs

    

    # throw dataframe with probs into MongoDB


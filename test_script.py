'''
This file will read in the full json file.
 & convert to a pandas dataframe.
Get a sample of one entry from the dataframe.
Determine if the entry is new.
If it is new, send the sample of one to the prediction.
 & Store the prediction in a MongoDB.
'''

import pandas as pd 
import numpy as np 
#import requests

from predict import predict

from pymongo import MongoClient
client = MongoClient('localhost', 27017)
db = client['frauds_test']
table = db['new_events_test']

def is_entry_new(df):
    '''
    Returns False if the entry is already in the mongodb
    '''
    new_id = str(df['object_id'])

    if len(list(table.find({'object_id': new_id}))) > 0:
        return False
    else:
        return True

def test_script():
    # read in full json file into pandas and convert to pandas
    df = pd.read_json('data/data.json')

    # Get a sample of one entry from the dataframe
    df_sample_1 = df.iloc[400, :].copy()

    # Determine if the entry is new
    new = is_entry_new(df_sample_1)

    # if it is new, send the sample to prediction
    if new:
        # get the prediction from predict.py
        df_1 = predict(df_sample_1)

        # create a dictionary that will be used to store items in mongodb
        d = {
            'object_id': str(df_1['object_id']),
            'name': df_1['name'],
            'org_name': df_1['org_name'],
            'fraud_probability': df_1['fraud_probability']
        }
        # store d into mongodb
        table.insert_one(d)

if __name__ == '__main__':
    test_script()


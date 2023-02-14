# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 13:21:42 2023

@author: walee
"""

import pandas as pd
import os

default_groundTruth = os.path.join('.','data','DBLP_Paper_Venue_FM_Literals2Nodes_SY1900_EY2021_50Class_GA_0_GSAINT_50_run2_output.csv')

""" Important Note: Make sure the prediction csv has 'paper' column that contains
    the URI of the paper."""
    
def verify(pred_file,true_file=default_groundTruth):
    pred_df = pd.read_csv(pred_file,)
    true_df = pd.read_csv(true_file,)
    
    # intersection_df = 
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:00:49 2023

@author: walee
"""

import pandas as pd
import os
import numpy as np

true_file= r'D:/Work/KGNET/GMLWebServiceApis/data/DBLP_Paper_Venue_FM_Literals2Nodes_SY1900_EY2021_50Class_GA_0_GSAINT_50_run2_output.csv'
pred_file = r'D:/Work/KGNET/GMLWebServiceApis/test_results.csv'
mismatch_file = r'D:/Work/KGNET/GMLWebServiceApis/data/mismatches.csv'
pred_df = pd.read_csv(pred_file,)
true_df = pd.read_csv(true_file,)

# mask_true = true_df['ent name'].isin(pred_df['paper'])
# inters_true = true_df[mask_true]
# mask_pred = pred_df['paper'].isin(inters_true['paper'])
# inters_pred = pred_df[mask_pred]

def markPreditions (pred_file,true_file,):
    pred_df = pd.read_csv(pred_file,)
    true_df = pd.read_csv(true_file,)
    intersection = pd.merge(true_df,pred_df,left_on='ent name' ,right_on='paper')
    pred_df['match_status'] = np.where(pred_df['venue'].isin(intersection['y_true']), 'POSITIVE', 'NEGATIVE')
    pred_df.to_csv(pred_file,index=False)

intersection = pd.merge(true_df,pred_df,left_on='ent name' ,right_on='paper')
pred_df['match_status'] = np.where(pred_df['venue'].isin(intersection['y_true']), 'match', 'mismatch')
# pred_df.to_csv(pred_file,index=False)


# intersection['match_status'] = np.where(intersection['venue'].isin(intersection['y_true']),'match','mismatch')
mismatch = intersection[intersection['match_status']=='mismatch']
mismatch = mismatch.drop(['y_true','ent name'],axis=1)
mismatch.to_csv(mismatch_file,index=False,encoding='utf-8')



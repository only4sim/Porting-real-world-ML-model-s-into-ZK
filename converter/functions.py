######################################################
# author: Devin Anzelmo, devinanzelmo@gmail.com
# licence: FreeBSD

"""
Copyright (c) 2015, Devin Anzelmo
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 OF THE POSSIBILITY OF SUCH DAMAGE.

"""


import pandas as pd
import numpy as np
import sys
sys.path.append('xgboost-0.40/wrapper/')
import xgboost as xgb

###############################################################################################

# takes the ceiling of the actual labels, aggregates values above the cutoff
def discretize_labels(labels,cutoff=69):
    new_labels = np.zeros(len(labels.index))
    for e,i in enumerate(labels.values.ravel()):
        if i > cutoff:
            new_labels[e] = 70
        elif i <= cutoff and i >= cutoff:
            new_labels[e] = cutoff
        else:
            new_labels[e] = np.ceil(i)
    return pd.DataFrame(np.abs(new_labels))

#Creates a xgboost object and loads model
def load_xgb_model(path_to_model):
    bst = xgb.Booster()
    bst.load_model(path_to_model)
    return bst

# loads the test data between lower and upper bound, 
def load_test_data(path_to_processed_data, lower_bound,upper_bound):
    full_test = pd.read_csv(path_to_processed_data + 'full_test.csv',index_col=0)
    hydro_test = pd.read_csv(path_to_processed_data + 'test_HydrometeorType_counts.csv',index_col=0)
    full_test = pd.concat([full_test, hydro_test],axis=1)

    # test_counts = pd.read_csv(path_to_processed_data + 'test_counts.csv',index_col=0,header=None)
    test_counts = pd.read_csv(path_to_processed_data + 'test_counts.csv',index_col=0)
    test_counts.columns = ['cnt']

    # train_counts = pd.read_csv(path_to_processed_data + 'train_counts.csv',index_col=0,header=None)
    train_counts = pd.read_csv(path_to_processed_data + 'train_counts.csv',index_col=0)
    train_counts.columns = ['cnt']

    # actual_labels = pd.read_csv(path_to_processed_data + 'actual_labels.csv',index_col=0,header=None)
    actual_labels = pd.read_csv(path_to_processed_data + 'actual_labels.csv',index_col=0)
    actual_labels.columns = ['label']

    discrete_labels = discretize_labels(actual_labels)
    discrete_labels.index = actual_labels.index
    discrete_labels.columns = ['d_lab']

    
    #now take just the parts needed for this problem
    if lower_bound == upper_bound:
        test_counts =  test_counts.query('cnt==@lower_bound')
        train_counts = train_counts.query('cnt==@lower_bound')
    else:
        test_counts =  test_counts.query('cnt > @lower_bound and cnt < @upper_bound')
        train_counts =  train_counts.query('cnt > @lower_bound and cnt < @upper_bound')

    discrete_labels = discrete_labels.reindex(train_counts.index)
    #get rid of all the labels with rain amount >=70
    discrete_labels_all = discrete_labels.copy()
    discrete_labels = discrete_labels.query('d_lab != 70')
    train_counts= train_counts.reindex(discrete_labels.index)
    actual_labels = actual_labels.reindex(discrete_labels.index)

    test = full_test.reindex(test_counts.index) 
    actual_labels = actual_labels.reindex(train_counts.index)

    to_drop = ['DistanceToRadar_' + x for x in ['sum','mad','sem','krt','skw','max','min','std','mean','med','num_non_null']] + [ 'HybridScan_num_00','HybridScan_num_03','RadarQualityIndex_num_99']
    test = test.drop(to_drop, axis=1)
    return test, discrete_labels,actual_labels, discrete_labels_all
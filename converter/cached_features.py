import json
import os
import pandas as pd
import numpy as np
import functions as fn

# Helper functions from the notebook
def test_col2i64_list_of_field(test_col):
    def num2i64_field(x):
        if x>=0:
            sgn = 1
        else:
            sgn = 0
        return f'{sgn}", "{abs(x)}'
    
    instr = '"'
    for i in test_col:
       # instr+= f'"{int(np.round(i*10000000000,0))}", '
        instr+=num2i64_field(int(np.round(i*10000000000,0)))
        instr+='", "'
    instr = instr[:-3]
    return instr

# Original functions from the notebook (modified to return both feature_names and instr)
def get_feature_names_instr_1():
    test, integer_labels, actual_labels, integer_labels_full = fn.load_test_data('../processed/',1,1)
    test = test.loc[:,test.mean() != -99999]
    feature_names = list(test.columns.to_numpy())
    instr = test_col2i64_list_of_field(test.iloc[1])
    return feature_names, instr

def get_feature_names_instr_2():
    test, integer_labels, actual_labels, integer_labels_full = fn.load_test_data('../processed/',1,4)
    feature_names = list(test.columns.to_numpy())
    instr = test_col2i64_list_of_field(test.iloc[1])
    return feature_names, instr

def get_feature_names_instr_3():
    test, integer_labels, actual_labels, integer_labels_full = fn.load_test_data('../processed/',3,8)
    feature_names = list(test.columns.to_numpy())
    instr = test_col2i64_list_of_field(test.iloc[1])
    return feature_names, instr

def get_feature_names_instr_4(use_xtra_features = True):
    test, integer_labels, actual_labels, integer_labels_full = fn.load_test_data('../processed/',7,18)

    if use_xtra_features:
        types = ['TimeToEnd','Reflectivity','Zdr','RR2','ReflectivityQC','RadarQualityIndex','RR3','RR1','Composite','RhoHV','HybridScan','LogWaterVolume']
        xtra_test = pd.DataFrame()
        for i in range(len(types)):
            xtra_test_temp = pd.read_csv('../processed/'+'test_'+types[i]+'8_17.csv',index_col=0)
            xtra_test = pd.concat([xtra_test,xtra_test_temp],axis=1)

        xtra_test = xtra_test.reindex(test.index)
        test= pd.concat([test, xtra_test],axis=1)
    feature_names = list(test.columns.to_numpy())
    instr = test_col2i64_list_of_field(test.iloc[1])
    return feature_names, instr

def get_feature_names_instr_5(use_xtra_features = True):
    test, integer_labels, actual_labels, integer_labels_full = fn.load_test_data('../processed/',17,1000)

    if use_xtra_features:
        types = ['TimeToEnd','Reflectivity','Zdr','RR2','ReflectivityQC','RadarQualityIndex','RR3','RR1','Composite','RhoHV','HybridScan','LogWaterVolume']
        xtra_test = pd.DataFrame()
        for i in range(len(types)):
            xtra_test_temp = pd.read_csv('../processed/'+'test_'+types[i]+'18_199.csv',index_col=0)
            xtra_test = pd.concat([xtra_test,xtra_test_temp],axis=1)

        xtra_test = xtra_test.reindex(test.index)
        test= pd.concat([test, xtra_test],axis=1)

    feature_names = list(test.columns.to_numpy())
    instr = test_col2i64_list_of_field(test.iloc[1])
    return feature_names, instr

def get_feature_names_instr(i, use_xtra_features):
    if i == 1:
        return get_feature_names_instr_1()
    elif i == 2:
        return get_feature_names_instr_2()
    elif i == 3:
        return get_feature_names_instr_3()
    elif i == 4:
        return get_feature_names_instr_4(use_xtra_features)
    elif i == 5:
        return get_feature_names_instr_5(use_xtra_features)
    else:
        print("Try to get features out of the scope.")
        return None, None

def save_feature_names_cache(cache_file='feature_names_cache.json'):
    """
    Generate and save feature names and instr for all models to a cache file using original functions.
    Run this once to create the cache.
    """
    cache = {}
    
    print("Generating feature names and instr cache (this may take a while)...")
    
    # Models 1-3 don't use extra features
    for model_num in range(1, 4):
        print(f"Processing model {model_num}...")
        feature_names, instr = get_feature_names_instr(model_num, False)
        cache[f'model_{model_num}'] = {
            'feature_names': feature_names,
            'instr': instr
        }
    
    # Models 4-5 have both with and without extra features
    for model_num in range(4, 6):
        print(f"Processing model {model_num} with extra features...")
        feature_names, instr = get_feature_names_instr(model_num, True)
        cache[f'model_{model_num}_with_xtra'] = {
            'feature_names': feature_names,
            'instr': instr
        }
        print(f"Processing model {model_num} without extra features...")
        feature_names, instr = get_feature_names_instr(model_num, False)
        cache[f'model_{model_num}_no_xtra'] = {
            'feature_names': feature_names,
            'instr': instr
        }
    
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)
    
    print(f"Feature names and instr cached to {cache_file}")
    return cache

def load_feature_names_from_cache(model_num, use_xtra_features=True, cache_file='feature_names_cache.json'):
    """
    Load feature names from cache file - extremely fast.
    
    Args:
        model_num: 1-5 for different models  
        use_xtra_features: Whether to include extra features (models 4&5 only)
        cache_file: Path to cache file
    
    Returns:
        List of feature names
    """
    if not os.path.exists(cache_file):
        print(f"Cache file {cache_file} not found. Creating cache...")
        save_feature_names_cache(cache_file=cache_file)
    
    with open(cache_file, 'r') as f:
        cache = json.load(f)
    
    if model_num in [4, 5]:
        key = f'model_{model_num}_{"with_xtra" if use_xtra_features else "no_xtra"}'
    else:
        key = f'model_{model_num}'
    
    return cache[key]['feature_names']

def load_instr_from_cache(model_num, use_xtra_features=True, cache_file='feature_names_cache.json'):
    """
    Load instruction vector from cache file - extremely fast.
    
    Args:
        model_num: 1-5 for different models  
        use_xtra_features: Whether to include extra features (models 4&5 only)
        cache_file: Path to cache file
    
    Returns:
        Instruction string for the model
    """
    if not os.path.exists(cache_file):
        print(f"Cache file {cache_file} not found. Creating cache...")
        save_feature_names_cache(cache_file=cache_file)
    
    with open(cache_file, 'r') as f:
        cache = json.load(f)
    
    if model_num in [4, 5]:
        key = f'model_{model_num}_{"with_xtra" if use_xtra_features else "no_xtra"}'
    else:
        key = f'model_{model_num}'
    
    return cache[key]['instr']

def load_feature_names_and_instr_from_cache(model_num, use_xtra_features=True, cache_file='feature_names_cache.json'):
    """
    Load both feature names and instruction vector from cache file - extremely fast.
    
    Args:
        model_num: 1-5 for different models  
        use_xtra_features: Whether to include extra features (models 4&5 only)
        cache_file: Path to cache file
    
    Returns:
        Tuple of (feature_names, instr)
    """
    if not os.path.exists(cache_file):
        print(f"Cache file {cache_file} not found. Creating cache...")
        save_feature_names_cache(cache_file=cache_file)
    
    with open(cache_file, 'r') as f:
        cache = json.load(f)
    
    if model_num in [4, 5]:
        key = f'model_{model_num}_{"with_xtra" if use_xtra_features else "no_xtra"}'
    else:
        key = f'model_{model_num}'
    
    return cache[key]['feature_names'], cache[key]['instr']
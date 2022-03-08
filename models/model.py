import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
tqdm.pandas()
import joblib
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import gmean
from scipy.stats import hmean
from scipy.stats import moment
from scipy.stats import variation
from lightgbm import LGBMRegressor
from sklearn.decomposition import PCA
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import NMF



# function to add 10 new features
def aggregate_row(row):
    '''Function to add new features with non zero stats'''

    # if the values are non zero then add new features
    non_zero_values = row.iloc[np.array(row).nonzero()].astype(float)
    if non_zero_values.empty:

        aggs = {
            'non_zero_mean': 0.0,
            'non_zero_std': 0.0,
            'non_zero_max': 0.0,
            'non_zero_min': 0.0,
            'non_zero_sum': 0.0,
            'non_zero_skewness': 0.0,
            'non_zero_kurtosis': 0.0,
            'non_zero_moment': 0.0,
            'non_zero_log_q1': 0.0,
            'non_zero_log_q3': 0.0
        }
    else:
        aggs = {
            'non_zero_mean': non_zero_values.mean(),
            'non_zero_std': non_zero_values.std(),
            'non_zero_max': non_zero_values.max(),
            'non_zero_min': non_zero_values.min(),
            'non_zero_sum': non_zero_values.sum(),
            'non_zero_skewness': skew(non_zero_values),
            'non_zero_kurtosis': kurtosis(non_zero_values),
            'non_zero_moment': moment(non_zero_values),
            'non_zero_log_q1': np.percentile(np.log1p(non_zero_values), q=25),
            'non_zero_log_q3': np.percentile(np.log1p(non_zero_values), q=75)
        }
    return pd.Series(aggs, index=list(aggs.keys()))



# preprocessing the data:
def preprocessing(test_data):
    features_test = test_data.drop(['ID', ], axis=1)
    features_test = np.log1p(features_test)

    # feature decomposition using PCA
    pca = PCA(n_components=20)
    features_test_PCA = pca.fit_transform(features_test)

    # feature decomposition using SRP
    srp = SparseRandomProjection(n_components=75, eps=0.28, dense_output=False)
    features_test_SRP = srp.fit_transform(features_test)

    # feature decomposition using NMF
    nmf = NMF(n_components=40, init=None, solver="cd", beta_loss="frobenius",
              tol=0.0001, max_iter=200, random_state=None, alpha=0.0,
              l1_ratio=0.0, verbose=0, shuffle=False)
    features_test_NMF = nmf.fit_transform(features_test)

    # concatenating the decomposed features all together
    features_test_PCA = features_test_PCA[:, :20]
    features_test_SRP = features_test_SRP[:, :75]
    features_test_NMF = features_test_NMF[:, :40]
    features_test_dec = np.hstack((features_test_SRP, features_test_PCA, features_test_NMF))

    # feature engineered for 10 statistical columns
    eng_features_test_ = features_test.iloc[:, :].progress_apply(aggregate_row, axis=1)

    # loading the column list pickle file
    col_init_test = joblib.load("col_init.pkl")
    col_reduce = [i for i in col_init_test if i in features_test.columns]
    features_test = features_test[col_reduce]
    print("Shape of the test features decomposed---", features_test.shape)

    ## concatinating the new 10 features
    features_test = pd.concat([features_test, eng_features_test_], axis=1)

    # concatenating the data with decomposition components
    features_test_dec = pd.DataFrame(features_test_dec)
    features_test = pd.concat([features_test, features_test_dec], axis=1)

    # loading the final column list pickle file
    col_final_test = joblib.load('col_final.pkl')
    features_test = features_test[col_final_test]

    return features_test



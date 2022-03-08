import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMRegressor



# function to make the predictions
def predict_value(features_test, test_data):
    # now loading the best model
    best_model_of_all = joblib.load('best_model.pkl')

    # making our prediciton
    y_hat = best_model_of_all.predict(features_test)

    # converting the values from log natural form to back
    y_pred = np.expm1(y_hat)

    # setting up the display format of our target variable
    pd.set_option('display.float_format', '{:.2f}'.format)

    # converting our prediction into dataframe
    pred_ = pd.DataFrame({'ID': test_data["ID"], 'Prediction': y_pred})

    return pred_


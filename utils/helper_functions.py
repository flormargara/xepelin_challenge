import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.statespace.tools import diff
from statsmodels.tsa.stattools import adfuller

def process_metrics(y_true, y_pred):
    """Compute metrics based on model predictions.
    Args:
        - y_true: ground truth.
        - y_pred: prediction.
    Returns:
        - metrics: dictionary with all computed metrics.
    """
    metrics = {}

    metrics['MAE'] = mean_absolute_error(y_true,y_pred)
    metrics['MSE'] = mean_squared_error(y_true,y_pred)
    metrics['RMSE'] = np.sqrt(mean_squared_error(y_true,y_pred))
    metrics['RMSLogE'] = np.log(np.sqrt(mean_squared_error(y_true,y_pred)))
    metrics['r2'] = r2_score(y_true,y_pred)
   # metrics['r2Adjust'] = 1 - ((1-r2_score(y_test,y_pred))*(n-1)/(n-k-1))

    return metrics
  
  
  def stationary_check(series, k_diff) -> str:
    """This function uses the Augmented DF test to check if
    time series is stationary"""
    
    ts = diff(series,k_diff = k_diff)
    res = adfuller(ts.dropna(),autolag='AIC')

    return f"P-Value is {res[1]}, therefore there's{' no' if res[1] > .05 else ''} evidence that supports the series is stationary"
  
  
  

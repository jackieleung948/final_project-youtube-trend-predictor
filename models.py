import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def forecast_arima(history, h, order=(1, 1, 1)):
    """
    Forecast the next value in the series using ARIMA model.
    
    Parameters:
    - history: pandas Series, the time series data
    - h: int, number of steps to forecast
    - order: tuple, the (p,d,q) order of the ARIMA model
    
    Returns:
    - forecast: numpy array, the forecasted values for the next h steps
    """
    try:
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=h)
        return np.clip(np.asarray(forecast), 0, None)  # Ensure non-negative forecasts
    except Exception:
        return None

def naive_last(history, h):
    """
    Forecast the next value in the series using the naive last value method.
    
    Parameters:
    - history: pandas Series, the time series data
    - h: int, number of steps to forecast
    
    Returns:
    - forecast: numpy array, the forecasted values for the next h steps
    """
    return np.repeat(float(np.asarrayhistory[-1]), h)  # Repeat the last observed value for h steps

def naive_mean(history, h):
    """
    Forecast the next value in the series using the naive mean method.
    
    Parameters:
    - history: pandas Series, the time series data
    - h: int, number of steps to forecast
    
    Returns:
    - forecast: numpy array, the forecasted values for the next h steps
    """
    return np.repeat(float(history.mean()), h)  # Repeat the mean of observed values for h steps
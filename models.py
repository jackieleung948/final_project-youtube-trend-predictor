import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

tf.random.set_seed(42)  # For reproducibility
np.random.seed(42)  # For reproducibility

N_STEPS = 6

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
    return np.repeat(float(np.asarray(history)[-1]), h)  # Repeat the last observed value for h steps

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

# Code from https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def fit_lstm(train, n_steps=N_STEPS):
    """
    Fit once per keyword on the training buckets.
    Return (model, low, high) or None if the series is too short or flat
    """
    train = np.asarray(train, dtype=float)
    low, high = train.min(), train.max()
    if high == low or len(train) <= n_steps + 1:
        return None
    scaled = (train - low) / (high - low)
    X, y = split_sequence(scaled, n_steps)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential([Input(shape=(n_steps, 1)), LSTM(50, activation="relu"), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)
    model.fit(X, y, epochs=200, verbose=0, callbacks=[es])
    return model, low, high

def forecast_lstm(fitted, history, h, n_steps=N_STEPS):
    """
    Recursive multi-step from the tail of history, using a model fit on train
    """
    if fitted is None:
        return None
    model, low, high = fitted
    sequence = ((np.asarray(history, dtype=float) - low) / (high - low)).tolist()[-n_steps:]
    out = []
    for _ in range(h):
        x = np.array(sequence[-n_steps:]).reshape((1, n_steps, 1))
        nxt = float(model.predict(x, verbose=0)[0][0])
        out.append(nxt); sequence.append(nxt)
    return np.clip(np.array(out) * (high - low) + low, 0, None)




    
import pandas as pd
from matplotlib import pyplot as plt
from numpy import int64
from datetime import datetime
from shared.secrets import API_KEY
from alpha_vantage.timeseries import TimeSeries


def fetch_api_data(company_code, period):
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    if period.lower() == "daily":
        data, meta_data = ts.get_daily_adjusted(symbol=company_code)
    elif period.lower() == "hourly":
        data, meta_data = ts.get_intraday(symbol=company_code)
    else:
        data, meta_data = ts.get_weekly(symbol=company_code)
    data.rename(columns=lambda x: x.split('. ')[-1], inplace=True)
    data.index = pd.to_datetime(data.index)
    data['Date'] = (data.index - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    data = data.sort_values('Date')

    return data, meta_data


def plot_predictions(x_test_1, y_test_1, y_pred_1, title):
    X_test_dates = pd.to_datetime(x_test_1.flatten(), unit='s')

    sorted_indices = X_test_dates.argsort()
    X_test_dates_sorted = X_test_dates.values[sorted_indices]
    y_test_sorted = y_test_1[sorted_indices.flatten()]
    y_pred_sorted = y_pred_1[sorted_indices.flatten()]

    plt.figure(figsize=(14, 8))
    plt.plot(X_test_dates_sorted, y_test_sorted, color='blue', label='Actual')
    plt.plot(X_test_dates_sorted, y_pred_sorted, color='red', label='Predicted')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()
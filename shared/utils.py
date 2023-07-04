import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from matplotlib import pyplot as plt

from shared.secrets import API_KEY


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

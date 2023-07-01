import pandas as pd
from flask import Flask, request
from numpy import int64

from shared.secrets import API_KEY

from alpha_vantage.timeseries import TimeSeries


app = Flask(__name__)


@app.route('/hello/<code>', methods=['GET'])
def index(code):
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    data, meta_data = ts.get_daily_adjusted(symbol=code)
    data.rename(columns=lambda x: x.split('. ')[-1], inplace=True)
    print(data)
    return data.to_json(orient='index')


if __name__ == '__main__':
    app.run()

from alpha_vantage.timeseries import TimeSeries
from flask import Flask, make_response

from models.linear_regression.linear_regression_main import linear_regression_prediction
from shared.secrets import API_KEY

app = Flask(__name__)


@app.route('/hello/<code>', methods=['GET'])
def index(code):
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    data, meta_data = ts.get_daily_adjusted(symbol=code)
    data.rename(columns=lambda x: x.split('. ')[-1], inplace=True)
    data = data[['close']]
    print(data['close'])
    return data.to_json(orient='index')


@app.route('/linear/<company_code>/<period>', methods=['GET'])
def linear_regression(company_code, period):
    body = linear_regression_prediction(company_code, period)
    return make_response(body, 200)


if __name__ == '__main__':
    app.run()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np


from models.prophet_models.prophet_linear_regression import ProphetLinearRegression
from shared.utils import fetch_api_data, plot_predictions, plot_test


def linear_regression_prediction(company_code, period):
    data, metadata = fetch_api_data(company_code, period)

    x_train_open, x_test_open, y_train_open, y_test_open = prepare_data(data, 'open')
    x_train_close, x_test_close, y_train_close, y_test_close = prepare_data(data, 'close')
    x_train_high, x_test_high, y_train_high, y_test_high = prepare_data(data, 'high')
    x_train_low, x_test_low, y_train_low, y_test_low = prepare_data(data, 'low')
    x_train_volume, x_test_volume, y_train_volume, y_test_volume = prepare_data(data, 'volume')

    y_pred_open, mse_open = predict(x_train_open, x_test_open, y_train_open, y_test_open)
    y_pred_close, mse_close = predict(x_train_close, x_test_close, y_train_close, y_test_close)
    y_pred_high, mse_high = predict(x_train_high, x_test_high, y_train_high, y_test_high)
    y_pred_low, mse_low = predict(x_train_low, x_test_low, y_train_low, y_test_low)
    y_pred_volume, mse_volume = predict(x_train_volume, x_test_volume, y_train_volume, y_test_volume)

    prophet_y_pred_open, prophet_mse_open = prophet_predict(x_train_open, x_test_open, y_train_open, y_test_open)
    prophet_y_pred_close, prophet_mse_close = prophet_predict(x_train_close, x_test_close, y_train_close, y_test_close)
    prophet_y_pred_high, prophet_mse_high = prophet_predict(x_train_high, x_test_high, y_train_high, y_test_high)
    prophet_y_pred_low, prophet_mse_low = prophet_predict(x_train_low, x_test_low, y_train_low, y_test_low)
    prophet_y_pred_volume, prophet_mse_volume = prophet_predict(x_train_volume, x_test_volume, y_train_volume, y_test_volume)

    response = {
        "companyCode": company_code,
        "period": period,
        "dates": x_test_open.flatten().tolist(),
        "open": {
            "predictedCustom": prophet_y_pred_open.tolist(),
            "predictedLibrary": y_pred_open.tolist(),
            "errorCustom": prophet_mse_open,
            "errorLibrary": mse_open,
            "actual": y_test_open.tolist()
        },
        "close": {
            "predictedCustom": prophet_y_pred_close.tolist(),
            "predictedLibrary": y_pred_close.tolist(),
            "errorCustom": prophet_mse_close,
            "errorLibrary": mse_close,
            "actual": y_test_close.tolist()
        },
        "high": {
            "predictedCustom": prophet_y_pred_high.tolist(),
            "predictedLibrary": y_pred_high.tolist(),
            "errorCustom": prophet_mse_high,
            "errorLibrary": mse_high,
            "actual": y_test_high.tolist()
        },
        "low": {
            "predictedCustom": prophet_y_pred_low.tolist(),
            "predictedLibrary": y_pred_low.tolist(),
            "errorCustom": prophet_mse_low,
            "errorLibrary": mse_low,
            "actual": y_test_low.tolist()
        },
        "volume": {
            "predictedCustom": prophet_y_pred_volume.tolist(),
            "predictedLibrary": y_pred_volume.tolist(),
            "errorCustom": prophet_mse_volume,
            "errorLibrary": mse_volume,
            "actual": y_test_volume.tolist()
        },
    }

    # return x_test_open, y_test_open, y_pred_open, prophet_y_pred_open
    return response


def predict(x_train, x_test, y_train, y_test):
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)

    return y_pred, mse


def prophet_predict(x_train, x_test, y_train, y_test):
    model = ProphetLinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)

    return y_pred, mse

def prepare_data(data, factor):
    y = data[factor].values
    x = data['Date'].values.reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Combine x_train and y_train
    combined_train = np.concatenate((x_train, y_train.reshape(-1, 1)), axis=1)

    # Sort the combined_train array by x_train
    sorted_combined_train = combined_train[combined_train[:, 0].argsort()]

    # Split the sorted_combined_train array into x_train and y_train
    x_train = sorted_combined_train[:, 0].reshape(-1, 1)
    y_train = sorted_combined_train[:, 1]

    # Combine x_test and y_test
    combined_test = np.concatenate((x_test, y_test.reshape(-1, 1)), axis=1)

    # Sort the combined_test array by x_test
    sorted_combined_test = combined_test[combined_test[:, 0].argsort()]

    # Split the sorted_combined_test array into x_test and y_test
    x_test = sorted_combined_test[:, 0].reshape(-1, 1)
    y_test = sorted_combined_test[:, 1]

    return x_train, x_test, y_train, y_test


# if __name__ == '__main__':
#     # x_test_1, y_test_1, y_pred_1, prophet_y_pred_1 = linear_regression_prediction("IBM", "weekly")
#     # plot_predictions(x_test_1, y_test_1, y_pred_1, prophet_y_pred_1, "Plot title")
#     plot_test(linear_regression_prediction("AMZN", "hourly"))

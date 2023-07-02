from models.prophet_models.prophet_decision_tree import MyDecisionTreeRegressor
from shared.utils import fetch_api_data
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np


def decision_tree_regression_prediction(company_code, period):
    data, metadata = fetch_api_data(company_code, period)

    x_train_open, x_test_open, y_train_open, y_test_open = prepare_data(data, 'open')
    x_train_close, x_test_close, y_train_close, y_test_close = prepare_data(data, 'close')
    x_train_high, x_test_high, y_train_high, y_test_high = prepare_data(data, 'high')
    x_train_low, x_test_low, y_train_low, y_test_low = prepare_data(data, 'low')
    x_train_volume, x_test_volume, y_train_volume, y_test_volume = prepare_data(data, 'volume')

    y_pred_open, mae_open = predict(x_train_open, x_test_open, y_train_open, y_test_open)
    y_pred_close, mae_close = predict(x_train_close, x_test_close, y_train_close, y_test_close)
    y_pred_high, mae_high = predict(x_train_high, x_test_high, y_train_high, y_test_high)
    y_pred_low, mae_low = predict(x_train_low, x_test_low, y_train_low, y_test_low)
    y_pred_volume, mae_volume = predict(x_train_volume, x_test_volume, y_train_volume, y_test_volume)

    prophet_y_pred_open, prophet_mae_open = prophet_predict(x_train_open, x_test_open, y_train_open, y_test_open)
    prophet_y_pred_close, prophet_mae_close = prophet_predict(x_train_close, x_test_close, y_train_close, y_test_close)
    prophet_y_pred_high, prophet_mae_high = prophet_predict(x_train_high, x_test_high, y_train_high, y_test_high)
    prophet_y_pred_low, prophet_mae_low = prophet_predict(x_train_low, x_test_low, y_train_low, y_test_low)
    prophet_y_pred_volume, prophet_mae_volume = prophet_predict(x_train_volume, x_test_volume, y_train_volume,
                                                                y_test_volume)

    response = {
        "companyCode": company_code,
        "period": period,
        "dates": x_test_open.flatten().tolist(),
        "open": {
            "predictedCustom": prophet_y_pred_open,
            "predictedLibrary": y_pred_open.tolist(),
            "errorCustom": prophet_mae_open,
            "errorLibrary": mae_open,
            "actual": y_test_open.tolist()
        },
        "close": {
            "predictedCustom": prophet_y_pred_close,
            "predictedLibrary": y_pred_close.tolist(),
            "errorCustom": prophet_mae_close,
            "errorLibrary": mae_close,
            "actual": y_test_close.tolist()
        },
        "high": {
            "predictedCustom": prophet_y_pred_high,
            "predictedLibrary": y_pred_high.tolist(),
            "errorCustom": prophet_mae_high,
            "errorLibrary": mae_high,
            "actual": y_test_high.tolist()
        },
        "low": {
            "predictedCustom": prophet_y_pred_low,
            "predictedLibrary": y_pred_low.tolist(),
            "errorCustom": prophet_mae_low,
            "errorLibrary": mae_low,
            "actual": y_test_low.tolist()
        },
        "volume": {
            "predictedCustom": prophet_y_pred_volume,
            "predictedLibrary": y_pred_volume.tolist(),
            "errorCustom": prophet_mae_volume,
            "errorLibrary": mae_volume,
            "actual": y_test_volume.tolist()
        },
    }

    return response


def predict(x_train, x_test, y_train, y_test):
    model = DecisionTreeRegressor()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)

    return y_pred, mae


def prophet_predict(x_train, x_test, y_train, y_test):
    model = MyDecisionTreeRegressor(max_depth=20)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)

    return y_pred, mae


def prepare_data(data, factor):
    y = data[factor].values
    x = data['Date'].values.reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    combined_train = np.concatenate((x_train, y_train.reshape(-1, 1)), axis=1)

    sorted_combined_train = combined_train[combined_train[:, 0].argsort()]

    x_train = sorted_combined_train[:, 0].reshape(-1, 1)
    y_train = sorted_combined_train[:, 1]

    combined_test = np.concatenate((x_test, y_test.reshape(-1, 1)), axis=1)

    sorted_combined_test = combined_test[combined_test[:, 0].argsort()]

    x_test = sorted_combined_test[:, 0].reshape(-1, 1)
    y_test = sorted_combined_test[:, 1]

    return x_train, x_test, y_train, y_test

from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np

from shared.utils import fetch_api_data


def support_vector_machine_prediction(company_code, period):
    data, metadata = fetch_api_data(company_code, period)

    x_train_open, x_test_open, y_train_open, y_test_open, scaler_x_open, scaler_y_open = prepare_data(data, 'open')
    x_train_close, x_test_close, y_train_close, y_test_close, scaler_x_close, scaler_y_close = prepare_data(data, 'close')
    x_train_high, x_test_high, y_train_high, y_test_high, scaler_x_high, scaler_y_high = prepare_data(data, 'high')
    x_train_low, x_test_low, y_train_low, y_test_low, scaler_x_low, scaler_y_low = prepare_data(data, 'low')
    x_train_volume, x_test_volume, y_train_volume, y_test_volume, scaler_x_volume, scaler_y_volume = prepare_data(data, 'volume')

    y_pred_open, mse_open = predict(x_train_open, x_test_open, y_train_open, y_test_open)
    y_pred_close, mse_close = predict(x_train_close, x_test_close, y_train_close, y_test_close)
    y_pred_high, mse_high = predict(x_train_high, x_test_high, y_train_high, y_test_high)
    y_pred_low, mse_low = predict(x_train_low, x_test_low, y_train_low, y_test_low)
    y_pred_volume, mse_volume = predict(x_train_volume, x_test_volume, y_train_volume, y_test_volume)

    prophet_y_pred_open, prophet_mse_open = predict(x_train_open, x_test_open, y_train_open, y_test_open)
    prophet_y_pred_close, prophet_mse_close = predict(x_train_close, x_test_close, y_train_close, y_test_close)
    prophet_y_pred_high, prophet_mse_high = predict(x_train_high, x_test_high, y_train_high, y_test_high)
    prophet_y_pred_low, prophet_mse_low = predict(x_train_low, x_test_low, y_train_low, y_test_low)
    prophet_y_pred_volume, prophet_mse_volume = predict(x_train_volume, x_test_volume, y_train_volume, y_test_volume)

    x_test_open = scaler_x_open.inverse_transform(x_test_open)

    y_pred_open = scaler_y_open.inverse_transform(y_pred_open)
    y_pred_close = scaler_y_close.inverse_transform(y_pred_close)
    y_pred_high = scaler_y_high.inverse_transform(y_pred_high)
    y_pred_low = scaler_y_low.inverse_transform(y_pred_low)
    y_pred_volume = scaler_y_volume.inverse_transform(y_pred_volume)

    prophet_y_pred_open = scaler_y_open.inverse_transform(prophet_y_pred_open)
    prophet_y_pred_close = scaler_y_close.inverse_transform(prophet_y_pred_close)
    prophet_y_pred_high = scaler_y_high.inverse_transform(prophet_y_pred_high)
    prophet_y_pred_low = scaler_y_low.inverse_transform(prophet_y_pred_low)
    prophet_y_pred_volume = scaler_y_volume.inverse_transform(prophet_y_pred_volume)

    y_test_open = scaler_y_open.inverse_transform(y_test_open)
    y_test_close = scaler_y_close.inverse_transform(y_test_close)
    y_test_high = scaler_y_high.inverse_transform(y_test_high)
    y_test_low = scaler_y_low.inverse_transform(y_test_low)
    y_test_volume = scaler_y_volume.inverse_transform(y_test_volume)

    response = {
        "companyCode": company_code,
        "period": period,
        "dates": x_test_open.flatten().tolist(),
        "open": {
            "predictedCustom": prophet_y_pred_open.flatten().tolist(),
            "predictedLibrary": y_pred_open.flatten().tolist(),
            "errorCustom": prophet_mse_open,
            "errorLibrary": mse_open,
            "actual": y_test_open.flatten().tolist()
        },
        "close": {
            "predictedCustom": prophet_y_pred_close.flatten().tolist(),
            "predictedLibrary": y_pred_close.flatten().tolist(),
            "errorCustom": prophet_mse_close,
            "errorLibrary": mse_close,
            "actual": y_test_close.flatten().tolist()
        },
        "high": {
            "predictedCustom": prophet_y_pred_high.flatten().tolist(),
            "predictedLibrary": y_pred_high.flatten().tolist(),
            "errorCustom": prophet_mse_high,
            "errorLibrary": mse_high,
            "actual": y_test_high.flatten().tolist()
        },
        "low": {
            "predictedCustom": prophet_y_pred_low.flatten().tolist(),
            "predictedLibrary": y_pred_low.flatten().tolist(),
            "errorCustom": prophet_mse_low,
            "errorLibrary": mse_low,
            "actual": y_test_low.flatten().tolist()
        },
        "volume": {
            "predictedCustom": prophet_y_pred_volume.flatten().tolist(),
            "predictedLibrary": y_pred_volume.flatten().tolist(),
            "errorCustom": prophet_mse_volume,
            "errorLibrary": mse_volume,
            "actual": y_test_volume.flatten().tolist()
        },
    }

    return response


def predict(x_train, x_test, y_train, y_test):
    model = LinearSVR()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    return y_pred.reshape(-1, 1), mse


def prepare_data(data, factor):
    y = data[factor].values.reshape(-1, 1)
    x = data['Date'].values.reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler_x = StandardScaler()
    x_train = scaler_x.fit_transform(x_train)
    x_test = scaler_x.fit_transform(x_test)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.fit_transform(y_test)

    sorted_indices = np.argsort(x_test[:, 0])
    x_test = x_test[sorted_indices]
    y_test = y_test[sorted_indices]

    return x_train, x_test, y_train, y_test, scaler_x, scaler_y

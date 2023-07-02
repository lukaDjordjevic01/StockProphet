import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from models.prophet_models.prophet_neural_network import ProphetNeuralNetwork
from shared.utils import fetch_api_data


def neural_network_prediction(company_code, period):
    data, metadata = fetch_api_data(company_code, period)

    x_train_open, x_test_open, y_train_open, y_test_open, scaler_open, scaler_y_open = prepare_data(data, 'open')
    x_train_close, x_test_close, y_train_close, y_test_close, scaler_close, scaler_y_close = prepare_data(data, 'close')
    x_train_high, x_test_high, y_train_high, y_test_high, scaler_high, scaler_y_high = prepare_data(data, 'high')
    x_train_low, x_test_low, y_train_low, y_test_low, scaler_low, scaler_y_low = prepare_data(data, 'low')
    x_train_volume, x_test_volume, y_train_volume, y_test_volume, scaler_volume, scaler_y_volume = prepare_data(data,
                                                                                                                'volume')

    model_open = create_model(x_train_open.shape[1])
    model_close = create_model(x_train_close.shape[1])
    model_high = create_model(x_train_high.shape[1])
    model_low = create_model(x_train_low.shape[1])
    model_volume = create_model(x_train_volume.shape[1])

    y_pred_open, mse_open = predict(model_open, x_train_open, x_test_open, y_train_open, y_test_open)
    y_pred_close, mse_close = predict(model_close, x_train_close, x_test_close, y_train_close, y_test_close)
    y_pred_high, mse_high = predict(model_high, x_train_high, x_test_high, y_train_high, y_test_high)
    y_pred_low, mse_low = predict(model_low, x_train_low, x_test_low, y_train_low, y_test_low)
    y_pred_volume, mse_volume = predict(model_volume, x_train_volume, x_test_volume, y_train_volume, y_test_volume)

    prophet_y_pred_open, prophet_mse_open = prophet_predict(x_train_open, x_test_open, y_train_open, y_test_open)
    prophet_y_pred_close, prophet_mse_close = prophet_predict(x_train_close, x_test_close, y_train_close, y_test_close)
    prophet_y_pred_high, prophet_mse_high = prophet_predict(x_train_high, x_test_high, y_train_high, y_test_high)
    prophet_y_pred_low, prophet_mse_low = prophet_predict(x_train_low, x_test_low, y_train_low, y_test_low)
    prophet_y_pred_volume, prophet_mse_volume = prophet_predict(x_train_volume, x_test_volume, y_train_volume,
                                                                y_test_volume)

    prophet_y_pred_open_rescaled = scaler_y_open.inverse_transform(prophet_y_pred_open)
    prophet_y_pred_close_rescaled = scaler_y_close.inverse_transform(prophet_y_pred_close)
    prophet_y_pred_high_rescaled = scaler_y_high.inverse_transform(prophet_y_pred_high)
    prophet_y_pred_low_rescaled = scaler_y_low.inverse_transform(prophet_y_pred_low)
    prophet_y_pred_volume_rescaled = scaler_y_volume.inverse_transform(prophet_y_pred_volume)

    y_pred_open_rescaled = scaler_y_open.inverse_transform(y_pred_open)
    y_pred_close_rescaled = scaler_y_close.inverse_transform(y_pred_close)
    y_pred_high_rescaled = scaler_y_high.inverse_transform(y_pred_high)
    y_pred_low_rescaled = scaler_y_low.inverse_transform(y_pred_low)
    y_pred_volume_rescaled = scaler_y_volume.inverse_transform(y_pred_volume)

    y_test_open_rescaled = scaler_y_open.inverse_transform(y_test_open)
    y_test_close_rescaled = scaler_y_close.inverse_transform(y_test_close)
    y_test_high_rescaled = scaler_y_high.inverse_transform(y_test_high)
    y_test_low_rescaled = scaler_y_low.inverse_transform(y_test_low)
    y_test_volume_rescaled = scaler_y_volume.inverse_transform(y_test_volume)

    x_test_open = scaler_open.inverse_transform(x_test_open)

    response = {
        "companyCode": company_code,
        "period": period,
        "dates": x_test_open.flatten().tolist(),
        "open": {
            "predictedCustom": prophet_y_pred_open_rescaled.flatten().tolist(),
            "predictedLibrary": y_pred_open_rescaled.flatten().tolist(),
            "errorCustom": prophet_mse_open,
            "errorLibrary": mse_open,
            "actual": y_test_open_rescaled.flatten().tolist()
        },
        "close": {
            "predictedCustom": prophet_y_pred_close_rescaled.flatten().tolist(),
            "predictedLibrary": y_pred_close_rescaled.flatten().tolist(),
            "errorCustom": prophet_mse_close,
            "errorLibrary": mse_close,
            "actual": y_test_close_rescaled.flatten().tolist()
        },
        "high": {
            "predictedCustom": prophet_y_pred_high_rescaled.flatten().tolist(),
            "predictedLibrary": y_pred_high_rescaled.flatten().tolist(),
            "errorCustom": prophet_mse_high,
            "errorLibrary": mse_high,
            "actual": y_test_high_rescaled.flatten().tolist()
        },
        "low": {
            "predictedCustom": prophet_y_pred_low_rescaled.flatten().tolist(),
            "predictedLibrary": y_pred_low_rescaled.flatten().tolist(),
            "errorCustom": prophet_mse_low,
            "errorLibrary": mse_low,
            "actual": y_test_low_rescaled.flatten().tolist()
        },
        "volume": {
            "predictedCustom": prophet_y_pred_volume_rescaled.flatten().tolist(),
            "predictedLibrary": y_pred_volume_rescaled.flatten().tolist(),
            "errorCustom": prophet_mse_volume,
            "errorLibrary": mse_volume,
            "actual": y_test_volume_rescaled.flatten().tolist()
        },
    }

    return response


def predict(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)

    return y_pred, mse


def prophet_predict(x_train, x_test, y_train, y_test):
    model = ProphetNeuralNetwork(x_train.shape[1], 50, 1)
    model.fit(x_train, y_train, epochs=3*len(x_train))
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)

    return y_pred, mse


def create_model(input_dim):
    model = Sequential()
    model.add(Dense(50, input_dim=input_dim, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def prepare_data(data, factor):
    y = data[factor].values.reshape(-1, 1)
    x = data['Date'].values.reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler_x = MinMaxScaler(feature_range=(0, 1))
    x_train = scaler_x.fit_transform(x_train)
    x_test = scaler_x.transform(x_test)

    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)

    sorted_indices = np.argsort(x_test[:, 0])
    x_test = x_test[sorted_indices]
    y_test = y_test[sorted_indices]

    return x_train, x_test, y_train, y_test, scaler_x, scaler_y

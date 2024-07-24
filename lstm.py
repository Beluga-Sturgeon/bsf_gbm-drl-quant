import pandas as pd
import numpy as np
from urllib.request import urlopen
import json
import certifi
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

import sys

def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)

def download(ticker: str):
    url = "https://financialmodelingprep.com/api/v3/historical-price-full"
    apikey = open(f"apikey", "r").readline().strip()
    if ticker.endswith("=X"):
        url += "/{}?apikey={}&from=2000-01-01".format("USD" + ticker[:-2], apikey)
    else:
        url += "/{}?apikey={}&from=2000-01-01".format(ticker, apikey)
    json_data = get_jsonparsed_data(url)
    
    if "historical" not in json_data:
        raise ValueError("No historical data found for ticker: {}".format(ticker))
    
    historical_data = json_data["historical"]
    df = pd.DataFrame(historical_data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df.sort_index()
    return df

def main(ticker: str):
    toDisplay = download(ticker)

    df = toDisplay.last('6M')
    y = df['close'].fillna(method='ffill')
    y = y.values.reshape(-1, 1)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(y)
    y = scaler.transform(y)

    # Generate the input and output sequences
    n_lookback = 60  # length of input sequences (lookback period)
    n_forecast = 30  # length of output sequences (forecast period)
    X = []
    Y = []

    for i in range(n_lookback, len(y) - n_forecast + 1):
        X.append(y[i - n_lookback: i])
        Y.append(y[i: i + n_forecast])
    X = np.array(X)
    Y = np.array(Y)

    # Fit the model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(n_forecast))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, Y, epochs=75, batch_size=32, verbose=0)

    # Generate the forecasts
    X_ = y[-n_lookback:]  # last available input sequence
    X_ = X_.reshape(1, n_lookback, 1)

    Y_ = model.predict(X_).reshape(-1, 1)
    Y_ = scaler.inverse_transform(Y_)

    # Organize the results in a data frame
    df_past = df[['close']].reset_index()
    df_past.rename(columns={'date': 'Date', 'close': 'Actual'}, inplace=True)
    df_past['Date'] = pd.to_datetime(df_past['Date'])
    df_past['Forecast'] = np.nan
    df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]

    df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
    df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
    df_future['Forecast'] = Y_.flatten()
    df_future['Actual'] = np.nan

    results = df_past.append(df_future).set_index('Date')
    results = results[results["Forecast"].notna()]
    results["Open"] = results["Forecast"]
    results["High"] = results["Forecast"]
    results["Low"] = results["Forecast"]
    resultsShifted = results.shift(-1)
    results["Close"] = resultsShifted["Open"]
    final = pd.concat([toDisplay, results], sort=False, join="inner")
    csv_df = final.to_csv()

    # Plotting the data
    plt.figure(figsize=(14, 7))
    plt.plot(final.index, final['close'], label='Actual')
    plt.plot(results.index, results['Forecast'], label='Forecast', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'{ticker} Price Prediction')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    ticker = sys.argv[1]

    main(ticker)
import numpy as np
import pandas as pd
from datetime import datetime
import requests
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt


# Retrieving data of BTC/USDT from binance
def request_data():
    url = 'https://api.binance.com/api/v3/klines'
    req_params = {'symbol': 'BTCUSDT', 'interval': '1h', 'limit': 1000}

    res_data = requests.get(url, params=req_params)
    data = res_data.json()

    df = pd.DataFrame(data)
    df = df.iloc[:, :6]
    df.columns = ['Kline open time', 'Open price', 'High price', 'Low price', 'Close price', 'Volume']

    return df


def convert_to_csv(df):
    # Show data in readable csv
    for item in df['Kline open time']:
        real_time = datetime.fromtimestamp(item/1000).strftime('%Y-%m-%d %H:%M:%S')
        df['Kline open time'] = df['Kline open time'].replace([item], real_time)

    df.to_csv('btcusdt.csv', encoding='utf-8')


def preprocess_data(df):
    # Remove missing values
    df.dropna(inplace=True)
    df = df[['Close price']]

    # Scale the data to be between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df)

    # Split the data into training and testing sets
    split = 0.75
    split_index = int(len(df_scaled) * split)

    train_data = df_scaled[:split_index]
    test_data = df_scaled[split_index:]

    return train_data, test_data, scaler


def build_model(train_data):
    # Create input and output data
    X_train, y_train = [], []
    for i in range(60, len(train_data)):
        X_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshape input data to be 3-dimensional
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    # Compile the model and train it
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    return model


def evaluate_model(test_data, scaler, model):
    n_steps = 60 # define the number of time steps here
    n_features = 1 # define the number of features (in this case, we only have the "Close" price)

    X_test = []
    y_test = []
    for i in range(n_steps, test_data.shape[0]):
        X_test.append(test_data[i-n_steps:i, 0])
        y_test.append(test_data[i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], n_features))

    y_test = np.array(y_test)
    y_test = np.reshape(y_test, (-1, 1))

    # Make predictions on the test data
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    # Calculate mean squared error and mean absolute error
    mse = np.mean(np.square(predictions - y_test))
    mae = np.mean(np.abs(predictions - y_test))

    return mse, mae, predictions


def plot_data(df, predictions):
    # Create a new DataFrame with the predicted data and the original data
    new_df = pd.DataFrame(predictions, columns=['Predictions'])
    new_df['Actual'] = df['Close price'].values[len(df) - len(predictions):]
    new_df = new_df[['Actual', 'Predictions']]

    # Plot the predicted data
    plt.plot(new_df.index, new_df['Actual'])
    plt.plot(new_df.index, new_df['Predictions'])
    plt.legend(['Actual', 'Predictions'])
    plt.title('Prediction for 1 hour ahead')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()


df = request_data()
convert_to_csv(df)
train_data, test_data, scaler = preprocess_data(df)
model = build_model(train_data)
mse, mae, predictions = evaluate_model(test_data, scaler, model)
plot_data(df, predictions)

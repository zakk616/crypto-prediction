import requests
import json
import pandas as pd
import datetime as dt
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from numpy import array

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')


coin_list = ["BTCUSDT"]
last_datetime = dt.datetime(2021, 11, 11)
print(last_datetime)
print(dt.datetime.now())
datetimetes = dt.datetime(2021, 5, 30, 5, 0, 0)
print(datetimetes)


def get_bybit_bars(symbol, interval, startTime, endTime):
    url = "https://api.bybit.com/public/linear/kline"

    startTime = str(int(startTime.timestamp()))
    endTime = str(int(endTime.timestamp()))

    req_params = {"symbol": symbol, 'interval': interval,
                  'from': startTime, 'to': endTime}

    df = pd.DataFrame(json.loads(requests.get(
        url, params=req_params).text)['result'])

    if (len(df.index) == 0):
        return None

    df.index = [dt.datetime.fromtimestamp(x) for x in df.open_time]

    return df


def create_csv(coin, last_datetime=last_datetime):

    df_list = []

    while True:
        print(last_datetime)
        new_df = get_bybit_bars(coin, 5, last_datetime, dt.datetime.now())
        if new_df is None:
            break
        df_list.append(new_df)
        last_datetime = max(new_df.index) + dt.timedelta(0, 1)
        time.sleep(1)

    df = pd.concat(df_list)
    x = df['start_at'].to_numpy()
    ylist = []
    for i in x:
        ylist.append(dt.datetime.fromtimestamp(
            i).strftime('%Y-%m-%d %H:%M:%S'))

    df['time'] = ylist
    df.to_csv(f'{coin}5m.csv', index=False)
    df = pd.read_csv('BTCUSDT5m.csv')
    df1 = df.reset_index()['close']
    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))
    training_size = int(len(df1)*0.65)
    test_size = len(df1)-training_size
    train_data, test_data = df1[0:training_size,
                                :], df1[training_size:len(df1), :1]

    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]  # i=0, 0,1,2,3-----99   100
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return numpy.array(dataX), numpy.array(dataY)

    # reshape into X=t,t+1,t+2,t+3 and Y=t+4
    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    model.fit(X_train, y_train, validation_data=(
        X_test, ytest), epochs=10, batch_size=64, verbose=1)
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    x_input = test_data[len(test_data) - 100:].reshape(1, -1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()
    lst_output = []
    n_steps = 100
    i = 0
    while(i < 30):
        if(len(temp_input) > 100):
            # print(temp_input)
            x_input = np.array(temp_input[1:])
            print("{} day input {}".format(i, x_input))
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))
            # print(x_input)
            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i, yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            # print(temp_input)
            lst_output.extend(yhat.tolist())
            i = i+1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i = i+1
    day_new = np.arange(1, 101)
    day_pred = np.arange(101, 131)
    df3 = df1.tolist()
    df3.extend(lst_output)
    df3 = scaler.inverse_transform(df3).tolist()
    print(len(df1))
    print(len(df3))
    plt.plot(df3[len(df3) - 30:])
    plt.savefig("D:/codeCase/Python/threadng/static/images/graph.png",
                bbox_inches="tight")
    plt.close("all")

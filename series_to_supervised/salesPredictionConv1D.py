# Sales Prediction using a Convolution Neural Network
import imp
from matplotlib import pyplot as plt, colors as clr
import numpy as np
from numpy import array, hstack
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import AveragePooling1D
from keras.models import load_model
from pandas import DataFrame, read_csv, to_datetime
import datetime
import pandas as pd
from pyparsing import col

from sklearn.preprocessing import OneHotEncoder
# split a univariate sequence into samples


def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


DAY_OF_WEEK = {
    "MONDAY": 0,
    "TUESDAY": 1,
    "WEDNESDAY": 2,
    "THURSDAY": 3,
    "FRIDAY": 4,
    "SATURDAY": 5,
    "SUNDAY": 6
}


def string_to_date(dt, format='%d.%m.%Y'):
    return datetime.datetime.strptime(dt, format)


def day_of_week(dt):
    return string_to_date(dt).weekday()


def month(dt):
    return string_to_date(dt).month


def load_data():
    categories = read_csv("../salesPredictionData/item_categories.csv")
    items = read_csv("../salesPredictionData/items.csv")
    sales = read_csv("../salesPredictionData/sales_train.csv", index_col=False)
    sales_20949 = sales[sales["item_id"] == 20949]
    # print(sales[["item_id"]].value_counts())
    # print(sales.head())
    plot_20949 = DataFrame()
    plot_20949["value"] = sales_20949["item_cnt_day"] 
    # print(plot_20949.head())
    plot_20949["date"] = to_datetime(sales_20949["date"], format='%d.%m.%Y')
    
    plot_20949.set_index(['date'], inplace=True)
    plot_20949 = plot_20949.sort_values(by='date', ascending=True)
    # print(plot_20949.head())

    df = plot_20949
    df.reset_index(inplace=True)
    df['year'] = [d.year for d in df.date]
    df['month'] = [d.strftime('%m') for d in df.date]
    years = df['year'].unique()
    #print(df[['month',"value",'year']])
   
    df1 = df.groupby(['month','year'])['value'].sum().reset_index()
    df1 = df1.sort_values(by="month", ascending=True)
   
    sub = df1[df1['year'] == 2013]
    # print(sub.head())
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    colors = ['b','g','r','c','m','y','k']
    color_index = 0
    # plotting for one year
    for year in years:
        sub = df1[df1['year'] == year]
        sub["month"] = sub["month"].astype(int)
        sub.set_index('month',inplace=True)
        sub1 = DataFrame()
        sub1['month'] = [i+1 for i in range(12)]
        
        sub1['value'] = [(sub['value'][i+1] if i+1 in sub.index else 0)  for i in range(12)]
         print(sub1)
        ax.plot(sub1['month'], sub1['value'], color=colors[color_index], linestyle='-', label=year)
        color_index = color_index + 1
        if(color_index>6):
            color_index = 0

    # some formatting
    ax.set_title('2013')
    ax.set_ylabel('Discharge (m3/s)')
    ax.set_xlabel('Month')
    ax.legend()
    ax.grid()
    # plt.plot(df1)
    # plt.gcf().autofmt_xdate()
    plt.show()
    return 
    shop_encoded = OneHotEncoder(
    ).fit_transform(sales_20949[["shop_id"]])
    # print(pd.DataFrame(shop_encoded.toarray()))
    # print(sales_20949.shape)
    # print(pd.DataFrame(shop_encoded.toarray()).shape)
    # r = pd.concat([sales_20949.reset_index(), pd.DataFrame(
    #     shop_encoded.toarray())], axis=1)
    
    # print(r.shape)
    # print(r.head())
    return
    # sales_20949 = pd.concat(
    #     [sales_20949, pd.DataFrame(shop_encoded.toarray())], axis=1)
    print(sales_20949.head())
    # sales_20949["shop_id"] = shop_encoded.toarray()
    sales_20949 = sales_20949.drop(["item_id"], axis=1)

    # sales_20949.reset_index()
    sales_20949["day_of_week"] = sales_20949.apply(
        lambda x: day_of_week(x.date), axis=1)
    sales_20949["month"] = sales_20949.apply(
        lambda x: month(x.date), axis=1)
    # print(sales_20949.head())
    # print(sales_20949.corr())
    # print(type(sales_20949.date[352379]))
    sales_20949["date"] = to_datetime(sales_20949["date"], format='%d.%m.%Y')
    # print(type(sales_20949["date"][352379]))
    # print(categories.head())
    # print(items.head())
    sales_20949.set_index(['date'], inplace=True)
    sales_20949 = sales_20949.sort_values(by='date', ascending=True)
    print(sales_20949.head())


def prepare_multivariate():
    # define input sequence
    in_seq1 = np.array([
        *[x for x in range(1000)], *[x for x in range(1000)], *[x for x in range(1000)]])
    in_seq2 = np.array([*[x for x in range(50, 1050)],
                        *[x for x in range(50, 1050)],
                        *[x for x in range(50, 1050)]])
    out_seq = np.array(array([in_seq1[i]+in_seq2[i]
                       for i in range(len(in_seq1))]))

    in_seq1 = in_seq1.reshape((len(in_seq1), 1))
    in_seq2 = in_seq2.reshape((len(in_seq2), 1))
    out_seq = out_seq.reshape((len(out_seq), 1))
    # horizontally stack columns
    dataset = hstack((in_seq1, in_seq2, out_seq))
    # choose a number of time steps
    n_steps = 3
    # convert into input/output
    X, y = split_sequences(dataset, n_steps)
    train_entries = 2900
    train_X, train_y = X[:train_entries, :], y[:train_entries]
    test_X, test_y = X[train_entries:, :], y[train_entries:]

    # the dataset knows the number of features, e.g. 2
    n_features = X.shape[2]
    # define model
    return n_steps, n_features, train_X, train_y, test_X, test_y

    # demonstrate prediction
    # yhat = model.predict(test_X, verbose=0)
    # print(yhat)
    # print(test_y)
    # plt.plot(yhat, label='yhat')
    # plt.plot(test_y, label='Y')
    # plt.legend()
    # plt.show()


def model_multivariate(n_steps, n_features, train_X, train_y, test_X, test_y):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu',
              input_shape=(n_steps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    history = model.fit(train_X, train_y, epochs=100,
                        validation_data=(test_X, test_y), verbose=1)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()
    return model


# n_steps, n_features, train_X, train_y, test_X, test_y = prepare_multivariate()
# model = model_multivariate(
#     n_steps, n_features, train_X, train_y, test_X, test_y)

# print("Saving model")
# model.save("cov1D_multivariate")

# print("Reconstructing model")
# reconstructed_model = load_model("cov1D_multivariate")

# yhat = reconstructed_model.predict(test_X, verbose=0)
# # print(yhat)
# # print(test_y)
# plt.plot(yhat, label='yhat')
# plt.plot(test_y, label='Y')
# plt.legend()
# plt.show()
load_data()

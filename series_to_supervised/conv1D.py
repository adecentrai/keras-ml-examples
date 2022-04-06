# cnn example for timeseries
import imp
from matplotlib import pyplot as plt
import numpy as np
from numpy import array, hstack
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import AveragePooling1D
from keras.models import load_model
# split a univariate sequence into samples


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def univariate():
    # define input sequence
    raw_seq = [i*10 for i in range(100)]
    # choose a number of time steps
    n_steps = 3
    # split into samples
    X, y = split_sequence(raw_seq, n_steps)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    # define model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu',
                     input_shape=(n_steps, n_features)))
    model.add(AveragePooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(60, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=1000, verbose=0)
    # demonstrate prediction
    x_input = array([499, 508, 523])
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    print(yhat)

# split a multivariate sequence into samples


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


def prepare_multivariate():
    # define input sequence
    in_seq1 = np.array([
        *[x for x in range(1000)], *[x for x in range(1000)], *[x for x in range(1000)]])
    in_seq2 = np.array([*[x for x in range(50, 1050)],
                        *[x for x in range(50, 1050)],
                        *[x for x in range(50, 1050)]
                        ])
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


n_steps, n_features, train_X, train_y, test_X, test_y = prepare_multivariate()
model = model_multivariate(
    n_steps, n_features, train_X, train_y, test_X, test_y)

print("Saving model")
model.save("cov1D_multivariate")

print("Reconstructing model")
reconstructed_model = load_model("cov1D_multivariate")

yhat = reconstructed_model.predict(test_X, verbose=0)
# print(yhat)
# print(test_y)
plt.plot(yhat, label='yhat')
plt.plot(test_y, label='Y')
plt.legend()
plt.show()

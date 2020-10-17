from run import read_from_file, save_to_file

import cv2

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, roc_curve, plot_roc_curve
from sklearn.model_selection import GridSearchCV

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, wrappers, optimizers, backend, callbacks, initializers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

import math

def retrieve_X_and_y(preprocessed_data):
    '''
      Assumes preprocessing() has been called e.g. frames are stored appropriately
    '''
    # 1 - extract X and y
    X = []
    y = []

    for (ROIS, faces, features, label) in preprocessed_data:
        X.append(features)
        y.append(label)

    # 2 - split into train/val/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=0)

    print("------------TRAIN/VAL/TEST SUMMARY-------------")
    print("X_train : {} y_train : {}".format(len(X_train), len(y_train)))
    print("X_val : {} y_val : {}".format(len(X_val), len(y_val)))
    print("X_test : {} y_test : {}".format(len(X_test), len(y_test)))

    # 3 - print off some of the fake frames + real frames
    return X_train, y_train, X_val, y_val, X_test, y_test


def slice_data(data):
    sliced_data = []

    for (ROIS, faces, features, label) in data:
        features = features[0:265]
        sliced_data.append((ROIS, faces, features, label))

    return sliced_data


def run_SVM(X_train, y_train, X_test, y_test):
    # Best Score : 56%
    # Best Params : C - 10, kernel - rbf
    # Acc : 51%

    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    standardScaler = StandardScaler()
    svc = SVC()

    pipe = Pipeline(steps=[('imputer', imputer),
                           ('standardScaler', standardScaler),
                           ('SVC', svc)])

    parameters = {
        'SVC__kernel': ('linear', 'rbf'),
        'SVC__C': [1, 5, 10, 15],
    }

    search = GridSearchCV(pipe, parameters, n_jobs=-1)

    search.fit(X_train, y_train)

    print("Best Score : {}".format(search.best_score_))
    print("Best Params : {}".format(search.best_params_))

    y_pred = search.predict(X_test)
    testing_accuracy = accuracy_score(y_test, y_pred)
    print("SVM ACC : {}".format(testing_accuracy))

def build_CNN(optimizer, init):
    model = models.Sequential()
    model.add(layers.Conv1D(32, 3, kernel_initializer=init,
                            activation='relu', input_shape=(265, 1)))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(64, 3, kernel_initializer=init, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(64, 3, kernel_initializer=init, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, kernel_initializer=init,  activation='sigmoid'))
    model.add(layers.Dense(1))
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


def run_CNN(X_train, y_train, X_val, y_val, X_test, y_test):
    X_train = np.expand_dims(X_train, axis=2)
    X_val = np.expand_dims(X_val, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    # for i in range(0, 10):
    #   callback = callbacks.EarlyStopping(monitor='loss', patience=50)
    #   model = build_CNN(optimizers.Adam(learning_rate=0.0001), 'he_normal')
    #   model.fit(X_train, y_train, callbacks=[callback], epochs= 10000, batch_size=100, verbose=0)
    #   test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
    #   print("Test Loss : {} Test Acc : {}".format(test_loss, test_acc))

    model = wrappers.scikit_learn.KerasClassifier(
        build_fn=build_CNN)

    opti = [
        optimizers.Adam(learning_rate=0.000001),
    ]
    init = ['he_normal']
    epochs = [10000]
    batches = [110]

    param_grid = dict(optimizer=opti, epochs=epochs,
                      batch_size=batches, init=init)

    callback = callbacks.EarlyStopping(monitor='loss', patience=50)

    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(X_train, y_train, callbacks=[callback])

    print("Best: {} using {}".format(
        grid_result.best_score_, grid_result.best_params_))

    model = grid.best_estimator_.model
    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

    print("Model Config : {}".format(model.optimizer.get_config()))
    print("Test Loss : {} Test Acc : {}".format(test_loss, test_acc))


def build_AlexNet(optimizer, init):
    model = models.Sequential()
    model.add(layers.Conv1D(32, 3, kernel_initializer=init,
                            activation='relu', input_shape=(265, 1)))
    
    model.add(layers.Conv1D(64, 3, kernel_initializer=init, activation='relu'))
    model.add(layers.MaxPooling1D(2))

    model.add(layers.Conv1D(64, 3, kernel_initializer=init, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    
    model.add(layers.Conv1D(64, 3, kernel_initializer=init, activation='relu'))
    model.add(layers.Conv1D(64, 3, kernel_initializer=init, activation='relu'))
    model.add(layers.Conv1D(64, 3, kernel_initializer=init, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    
    model.add(layers.Dense(64, kernel_initializer=init,  activation='relu'))
    model.add(layers.Dense(64, kernel_initializer=init,  activation='relu'))

    # what happens if you remove the flatten??
    model.add(layers.Flatten())
    model.add(layers.Dense(64, kernel_initializer=init,  activation='sigmoid'))
    model.add(layers.Dense(1))

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

def run_AlexNet(X_train, y_train, X_val, y_val, X_test, y_test):
    X_train = np.expand_dims(X_train, axis=2)
    X_val = np.expand_dims(X_val, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    model = wrappers.scikit_learn.KerasClassifier(
        build_fn=build_AlexNet)

    he_avg_init = initializers.VarianceScaling(scale=2., mode='fan_avg', distribution="uniform")

    opti = [
        optimizers.Adam(learning_rate=0.00001),
        optimizers.Adam(learning_rate=0.000001),

    ]
    init = ['he_normal', he_avg_init]
    epochs = [10000]
    batches = [40, 50, 60, 60]

    param_grid = dict(optimizer=opti, epochs=epochs,
                      batch_size=batches, init=init)

    callback = callbacks.EarlyStopping(monitor='loss', patience=100)

    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(X_train, y_train, callbacks=[callback])

    print("Best: {} using {}".format(
        grid_result.best_score_, grid_result.best_params_))

    model = grid.best_estimator_.model
    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

    print("Model Config : {}".format(model.optimizer.get_config()))
    print("Test Loss : {} Test Acc : {}".format(test_loss, test_acc))


def build_CNN_RNN(optimizer, init):
    model = models.Sequential()

    model.add(layers.Conv1D(filters=20, kernel_size=4, strides=2, padding="valid", input_shape=(265, 1)))
    model.add(layers.GRU(20, return_sequences=True))
    model.add(layers.GRU(20))

    model.add(layers.Dense(20, kernel_initializer=init,  activation='sigmoid'))
    model.add(layers.Dense(1))

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


def run_CNN_RNN(X_train, y_train, X_val, y_val, X_test, y_test):
    X_train = np.expand_dims(X_train, axis=2)
    X_val = np.expand_dims(X_val, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    model = wrappers.scikit_learn.KerasClassifier(
        build_fn=build_CNN_RNN)

    he_avg_init = initializers.VarianceScaling(scale=2., mode='fan_avg', distribution="uniform")

    opti = [
        # optimizers.Adam(learning_rate=10),
        # optimizers.Adam(learning_rate=1),
        # optimizers.Adam(learning_rate=0.1),
        # optimizers.Adam(learning_rate=0.01),
        optimizers.Adam(learning_rate=0.0001),
        optimizers.Adam(learning_rate=0.00001),
    ]
    init = ['he_normal', he_avg_init]
    epochs = [10000]
    batches = [25, 50, 150]

    param_grid = dict(optimizer=opti, epochs=epochs,
                      batch_size=batches, init=init)

    callback = callbacks.EarlyStopping(monitor='loss', patience=100)

    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(X_train, y_train, callbacks=[callback])

    print("Best: {} using {}".format(
        grid_result.best_score_, grid_result.best_params_))

    model = grid.best_estimator_.model
    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

    print("Model Config : {}".format(model.optimizer.get_config()))
    print("Test Loss : {} Test Acc : {}".format(test_loss, test_acc))


def build_WaveNet(optimizer, init):
    model = models.Sequential()

    model.add(layers.InputLayer(input_shape=(265, 1)))

    for rate in (1, 2, 4, 8) * 2:
      model.add(layers.Conv1D(filters=20, kernel_size=2, activation="relu", dilation_rate=rate))
    
    model.add(layers.Conv1D(filters=10, kernel_size=1))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, kernel_initializer=init,  activation='sigmoid'))
    model.add(layers.Dense(1))

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', metrics=['accuracy'])

    return model


def run_WaveNet(X_train, y_train, X_val, y_val, X_test, y_test):
    X_train = np.expand_dims(X_train, axis=2)
    X_val = np.expand_dims(X_val, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    model = wrappers.scikit_learn.KerasClassifier(
        build_fn=build_WaveNet)

    he_avg_init = initializers.VarianceScaling(scale=2., mode='fan_avg', distribution="uniform")

    opti = [
        optimizers.Adam(learning_rate=10),
        optimizers.Adam(learning_rate=1),
        optimizers.Adam(learning_rate=0.1),
        optimizers.Adam(learning_rate=0.01),
        optimizers.Adam(learning_rate=0.0001),
        optimizers.Adam(learning_rate=0.00001),
    ]
    init = ['he_normal', he_avg_init]
    epochs = [10000]
    batches = [25, 50, 150]

    param_grid = dict(optimizer=opti, epochs=epochs,
                      batch_size=batches, init=init)

    callback = callbacks.EarlyStopping(monitor='loss', patience=100)

    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(X_train, y_train, callbacks=[callback])

    print("Best: {} using {}".format(
        grid_result.best_score_, grid_result.best_params_))

    model = grid.best_estimator_.model
    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

    print("Model Config : {}".format(model.optimizer.get_config()))
    print("Test Loss : {} Test Acc : {}".format(test_loss, test_acc))


def build_deep_WaveNet(optimizer, init):
    model = models.Sequential()

    model.add(layers.InputLayer(input_shape=(265, 1)))

    for rate in (1, 2, 4, 8, 16, 32, 64) * 2:
      model.add(layers.Conv1D(filters=20, kernel_size=2, activation="relu", dilation_rate=rate))
    
    model.add(layers.Conv1D(filters=10, kernel_size=1))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, kernel_initializer=init,  activation='sigmoid'))
    model.add(layers.Dense(1))

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', metrics=['accuracy'])

    return model

def run_deep_WaveNet(X_train, y_train, X_val, y_val, X_test, y_test):
    X_train = np.expand_dims(X_train, axis=2)
    X_val = np.expand_dims(X_val, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    model = wrappers.scikit_learn.KerasClassifier(
        build_fn=build_deep_WaveNet)

    he_avg_init = initializers.VarianceScaling(scale=2., mode='fan_avg', distribution="uniform")

    opti = [
        optimizers.Adam(learning_rate=10),
        optimizers.Adam(learning_rate=1),
        optimizers.Adam(learning_rate=0.1),
        optimizers.Adam(learning_rate=0.01),
        optimizers.Adam(learning_rate=0.0001),
        optimizers.Adam(learning_rate=0.00001),
    ]
    init = ['he_normal', he_avg_init]
    epochs = [10000]
    batches = [25, 50, 150]

    param_grid = dict(optimizer=opti, epochs=epochs,
                      batch_size=batches, init=init)

    callback = callbacks.EarlyStopping(monitor='loss', patience=50)

    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(X_train, y_train, callbacks=[callback])

    print("Best: {} using {}".format(
        grid_result.best_score_, grid_result.best_params_))

    model = grid.best_estimator_.model
    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

    print("Model Config : {}".format(model.optimizer.get_config()))
    print("Test Loss : {} Test Acc : {}".format(test_loss, test_acc))

def build_RNN(optimizer, init):
    # Try LSTM's first, then try GRU's next...
    model = models.Sequential()

    model.add(layers.LSTM(20, return_sequences=True, input_shape=(265, 1)))
    model.add(layers.LSTM(20, return_sequences=True))
    model.add(layers.LSTM(20))

    model.add(layers.Dense(20, kernel_initializer=init,  activation='sigmoid'))
    model.add(layers.Dense(1))

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


def run_RNN(X_train, y_train, X_val, y_val, X_test, y_test):
    X_train = np.expand_dims(X_train, axis=2)
    X_val = np.expand_dims(X_val, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    model = wrappers.scikit_learn.KerasClassifier(
        build_fn=build_RNN)

    he_avg_init = initializers.VarianceScaling(scale=2., mode='fan_avg', distribution="uniform")

    opti = [
        optimizers.Adam(learning_rate=1),
        optimizers.Adam(learning_rate=0.01),
        optimizers.Adam(learning_rate=0.0001),
        optimizers.Adam(learning_rate=0.00001),
    ]
    init = ['he_normal', he_avg_init]
    epochs = [10000]
    batches = [25, 50, 100, 150]

    param_grid = dict(optimizer=opti, epochs=epochs,
                      batch_size=batches, init=init)

    callback = callbacks.EarlyStopping(monitor='loss', patience=100)

    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(X_train, y_train, callbacks=[callback])

    print("Best: {} using {}".format(
        grid_result.best_score_, grid_result.best_params_))

    model = grid.best_estimator_.model
    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

    print("Model Config : {}".format(model.optimizer.get_config()))
    print("Test Loss : {} Test Acc : {}".format(test_loss, test_acc))

def build_best_WaveNet():
    he_avg_init = initializers.VarianceScaling(scale=2., mode='fan_avg', distribution="uniform")

    model = models.Sequential()

    model.add(layers.InputLayer(input_shape=(265, 1)))

    for rate in (1, 2, 4, 8) * 2:
      model.add(layers.Conv1D(filters=20, kernel_size=2, activation="relu", dilation_rate=rate))
    
    model.add(layers.Conv1D(filters=10, kernel_size=1))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, kernel_initializer=he_avg_init,  activation='sigmoid'))
    model.add(layers.Dense(1))

    return model

def evaluate_model(model, X_test, y_test):
    # X_test = np.asarray(X_test)
    # X_test = np.expand_dims(X_test, axis=2)

    _, test_acc = model.evaluate(X_test, y_test)
    print("VALIDATION ACC : ", test_acc)

    clf_probs = model.predict_proba(X_test)
    score = log_loss(y_test, clf_probs)
    print("LOG LOSS SCORE : ", score)

def save_keras_model_to_file(model, filename):
    model.save(filename)

def retrieve_keras_model_from_file(filename):
    return load_model(filename)

def run_best_WaveNet(X_train, y_train, X_val, y_val, X_test, y_test):
    X_train = np.asarray(X_train)
    X_train = np.expand_dims(X_train, axis=2)

    X_test = np.asarray(X_test)
    X_test = np.expand_dims(X_test, axis=2)

    y_train = np.asarray(y_train)

    model = build_best_WaveNet()

    opt = optimizers.Adam(learning_rate=0.00001)
    model.compile(loss = "binary_crossentropy", optimizer = opt, metrics=['accuracy'])

    es = EarlyStopping(monitor='loss', patience=100)

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10000, shuffle=True, callbacks=[es], batch_size=150)

    evaluate_model(model, X_test, y_test)
    save_keras_model_to_file(model, "wavenet.h5")
    return model

if __name__ == "__main__":
    data = read_from_file("E:\signal_data")
    data = slice_data(data)
    X_train, y_train, X_val, y_val, X_test, y_test = retrieve_X_and_y(data)

    # run_SVM(X_train, y_train, X_test, y_test)
    # run_CNN(X_train, y_train, X_val, y_val, X_test, y_test)
    # run_AlexNet(X_train, y_train, X_val, y_val, X_test, y_test)
    # run_CNN_RNN(X_train, y_train, X_val, y_val, X_test, y_test)
    # run_WaveNet(X_train, y_train, X_val, y_val, X_test, y_test)
    # run_RNN(X_train, y_train, X_val, y_val, X_test, y_test)
    # run_deep_WaveNet(X_train, y_train, X_val, y_val, X_test, y_test)
    model = run_best_WaveNet(X_train, y_train, X_val, y_val, X_test, y_test)
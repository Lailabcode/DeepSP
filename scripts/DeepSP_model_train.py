# Import libraries
import numpy as np
import pandas as pd
import random


# Import machine learning libraries
import tensorflow as tf
import keras
from keras.models import model_from_json
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)


def load_input_data(filename):
    name_list = []
    seq_list = []
    score_list = []

    with open(filename) as datafile:
        for line in datafile:
            line = line.strip().split()
            name_list.append(line[0])
            seq_list.append(line[1])
            score_temp = []
            for i in range(len(line[2:])):
                data = float(line[i + 2])
                score_temp.append(data)
            score_list.append(score_temp)
    return name_list, seq_list, score_list


def one_hot_encoder(s):
    d = {
        "A": 0,
        "C": 1,
        "D": 2,
        "E": 3,
        "F": 4,
        "G": 5,
        "H": 6,
        "I": 7,
        "K": 8,
        "L": 9,
        "M": 10,
        "N": 11,
        "P": 12,
        "Q": 13,
        "R": 14,
        "S": 15,
        "T": 16,
        "V": 17,
        "W": 18,
        "Y": 19,
        "-": 20,
    }

    x = np.zeros((len(d), len(s)))
    x[[d[c] for c in s], range(len(s))] = 1

    return x


def best_model_SAPpos():
    best_model = keras.Sequential(name="model_conv1D")

    best_model.add(keras.layers.Input(shape=(272, 21)))

    best_model.add(
        keras.layers.Conv1D(
            filters=128, kernel_size=5, activation="relu", name="Conv1D_1"
        )
    )
    best_model.add(BatchNormalization())
    best_model.add(keras.layers.Dropout(0.3))

    best_model.add(
        keras.layers.Conv1D(
            filters=96, kernel_size=4, activation="relu", name="Conv1D_2"
        )
    )
    best_model.add(BatchNormalization())

    best_model.add(
        keras.layers.Conv1D(
            filters=32, kernel_size=5, activation="relu", name="Conv1D_3"
        )
    )
    best_model.add(BatchNormalization())

    best_model.add(keras.layers.MaxPooling1D(pool_size=2, name="MaxPooling1D"))
    best_model.add(keras.layers.Flatten())

    # Input layer and First hidden layer of neural network
    best_model.add(keras.layers.Dense(units=112, activation="relu", name="Dense_1"))
    best_model.add(keras.layers.Dense(units=48, activation="relu", name="Dense_2"))
    best_model.add(keras.layers.Dense(10, name="Dense_3"))

    return best_model


def best_model_SCMpos():
    best_model = keras.Sequential(name="model_conv1D")

    best_model.add(keras.layers.Input(shape=(272, 21)))

    best_model.add(
        keras.layers.Conv1D(
            filters=128, kernel_size=4, activation="relu", name="Conv1D_1"
        )
    )
    best_model.add(BatchNormalization())
    best_model.add(keras.layers.Dropout(0.4))

    best_model.add(
        keras.layers.Conv1D(
            filters=112, kernel_size=4, activation="relu", name="Conv1D_2"
        )
    )
    best_model.add(BatchNormalization())
    best_model.add(keras.layers.Dropout(0.4))

    best_model.add(
        keras.layers.Conv1D(
            filters=144, kernel_size=5, activation="relu", name="Conv1D_3"
        )
    )
    best_model.add(BatchNormalization())
    best_model.add(keras.layers.Dropout(0.0))

    best_model.add(keras.layers.MaxPooling1D(pool_size=2, name="MaxPooling1D"))
    best_model.add(keras.layers.Flatten())

    # Input layer and First hidden layer of neural network
    best_model.add(keras.layers.Dense(units=128, activation="relu", name="Dense_1"))
    best_model.add(keras.layers.Dense(10, name="Dense_2"))

    return best_model


def best_model_SCMneg():
    best_model = keras.Sequential(name="model_conv1D")

    best_model.add(keras.layers.Input(shape=(272, 21)))

    best_model.add(
        keras.layers.Conv1D(
            filters=128, kernel_size=5, activation="relu", name="Conv1D_1"
        )
    )
    best_model.add(BatchNormalization())
    best_model.add(keras.layers.Dropout(0.1))

    best_model.add(
        keras.layers.Conv1D(
            filters=112, kernel_size=4, activation="relu", name="Conv1D_2"
        )
    )
    best_model.add(BatchNormalization())

    best_model.add(
        keras.layers.Conv1D(
            filters=64, kernel_size=4, activation="relu", name="Conv1D_3"
        )
    )
    best_model.add(BatchNormalization())

    best_model.add(keras.layers.MaxPooling1D(pool_size=2, name="MaxPooling1D"))
    best_model.add(keras.layers.Flatten())

    # Input layer and First hidden layer of neural network
    best_model.add(keras.layers.Dense(units=128, activation="relu", name="Dense_1"))
    best_model.add(keras.layers.Dense(10, name="Dense_2"))

    return best_model


# ts = 0.2; bs = 64
filenames = ["Deep_SAPpos_data.txt", "Deep_SCMpos_data.txt", "Deep_SCMneg_data.txt"]
models = [best_model_SAPpos(), best_model_SCMpos(), best_model_SCMneg()]
l_rates = [0.0001, 0.005, 0.0001]

for file, model, l_rate in zip(filenames, models, l_rates):
    prop = file.split("_")[1]

    name_list, seq_list, score_list = load_input_data("../data/" + file)
    X = seq_list
    y = score_list

    # Train and compile model with best hyperparameters
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=0
    )

    X_train = [one_hot_encoder(s=x) for x in X_train]
    X_train = np.transpose(np.asarray(X_train), (0, 2, 1))
    X_train = np.asarray(X_train)

    X_test = [one_hot_encoder(s=x) for x in X_test]
    X_test = np.transpose(np.asarray(X_test), (0, 2, 1))
    X_test = np.asarray(X_test)

    X_val = [one_hot_encoder(s=x) for x in X_val]
    X_val = np.transpose(np.asarray(X_val), (0, 2, 1))
    X_val = np.asarray(X_val)

    y_train = np.asarray(y_train).reshape((-1, 10))
    y_test = np.asarray(y_test).reshape((-1, 10))
    y_val = np.asarray(y_val).reshape((-1, 10))

    optimizer = Adam(learning_rate=l_rate)
    best_model = model
    best_model.compile(optimizer=optimizer, loss="mae", metrics=None)

    # Create callback
    filepath = "../data/Conv1D_regression_" + prop + ".h5"
    checkpoint = ModelCheckpoint(
        filepath=filepath,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
    )
    callbacks = [checkpoint]

    # Fit the CNN to the training set
    history = best_model.fit(
        x=X_train,
        y=y_train,
        shuffle=True,
        validation_data=(X_val, y_val),
        epochs=50,
        callbacks=callbacks,
        batch_size=64,
        verbose=2,
    )

    # Save the Conv1D architecture to json
    Conv1D_regression_json = best_model.to_json()
    with open("../data/Conv1D_regression" + prop + ".json", "w") as json_file:
        json_file.write(Conv1D_regression_json)

    # Load the Conv1D architecture from json
    pred_model = model_from_json(Conv1D_regression_json)

    # Load weights from the best model into Conv1D model
    pred_model.load_weights(filepath)

    # Compile the loaded Conv1D model
    pred_model.compile(optimizer=optimizer, metrics=["mae"])

    y_pred = pred_model.predict(X_test)

    best_val_loss = min(history.history["val_loss"])

    # Initialize lists to store baseline MAE and mean scores for each target
    baseline_mae_list = []
    mean_score_list = []

    for i in range(y_test.shape[1]):
        # Calculate the baseline MAE for the i-th target
        baseline_prediction = np.full_like(y_test[:, i], np.mean(y_test[:, i]))
        baseline_mae = mean_absolute_error(y_test[:, i], baseline_prediction)
        baseline_mae_list.append(baseline_mae)

        # Calculate the mean score for the i-th target
        mean_score = np.mean(y_test[:, i])
        mean_score_list.append(mean_score)

    # Initialize lists to store metrics for each target
    mae_list = []
    corr_list = []

    for i in range(y_test.shape[1]):
        # Calculate MAE for the i-th target
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        mae_list.append(mae)

        # Calculate correlation coefficient (correlation) for the i-th target
        corr = np.corrcoef(y_test[:, i], y_pred[:, i])[0, 1]
        corr_list.append(corr)

    reg = [
        "CDRH1",
        "CDRH2",
        "CDRH3",
        "CDRL1",
        "CDRL2",
        "CDRL3",
        "CDR",
        "Hv",
        "Lv",
        "Fv",
    ]

    result_dict = {
        "prop": [],
        "Mean_score": [],
        "Baseline_MAE": [],
        "Val_loss": [],
        "MAE": [],
        "R": [],
    }

    for r, i, j, k, l in zip(
        reg, mean_score_list, baseline_mae_list, mae_list, corr_list
    ):
        # Append the corresponding values to the result_dict
        result_dict["prop"].append(prop + r)
        result_dict["Mean_score"].append(i)
        result_dict["Baseline_MAE"].append(j)
        result_dict["Val_loss"].append(best_val_loss)
        result_dict["MAE"].append(k)
        result_dict["R"].append(l)

    # Create the DataFrame
    result_df = pd.DataFrame(result_dict)

    # Save the DataFrame to CSV
    result_df.to_csv("../data/hyp_metric_" + prop + ".csv", index=False)

    his_df = pd.DataFrame(history.history)
    his_df.to_csv("his" + prop + ".csv", index=False)

data_frames = []
for file in filenames:
    prop = file.split("_")[1]
    infile = "hyp_metric_" + prop + ".csv"
    df = pd.read_csv(infile)
    data_frames.append(df)
concatenated_df = pd.concat(data_frames, ignore_index=True)
concatenated_df.to_csv("../data/Final_model_metric.csv", index=False)

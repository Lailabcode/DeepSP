# Import libraries
import numpy as np
import pandas as pd
import random
import os # For TF environment variables
from pathlib import Path # For path management

# Import machine learning libraries
import tensorflow as tf
# import keras # Keras is now part of TensorFlow
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import BatchNormalization, Input, Conv1D, Dropout, MaxPooling1D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Import from our utility script
from utils import get_project_root, load_input_data_for_training, one_hot_encoder

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # Should be set before TF import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)


# one_hot_encoder is now imported from utils
# load_input_data is now imported from utils as load_input_data_for_training

def best_model_SAPpos():
    best_model = Sequential(name="model_conv1D_SAPpos")
    best_model.add(Input(shape=(272, 21)))
    best_model.add(Conv1D(filters=128, kernel_size=5, activation="relu", name="Conv1D_1"))
    best_model.add(BatchNormalization())
    best_model.add(Dropout(0.3))
    best_model.add(Conv1D(filters=96, kernel_size=4, activation="relu", name="Conv1D_2"))
    best_model.add(BatchNormalization())
    best_model.add(Conv1D(filters=32, kernel_size=5, activation="relu", name="Conv1D_3"))
    best_model.add(BatchNormalization())
    best_model.add(MaxPooling1D(pool_size=2, name="MaxPooling1D"))
    best_model.add(Flatten())
    best_model.add(Dense(units=112, activation="relu", name="Dense_1"))
    best_model.add(Dense(units=48, activation="relu", name="Dense_2"))
    best_model.add(Dense(10, name="Dense_3"))
    return best_model


def best_model_SCMpos():
    best_model = Sequential(name="model_conv1D_SCMpos")
    best_model.add(Input(shape=(272, 21)))
    best_model.add(Conv1D(filters=128, kernel_size=4, activation="relu", name="Conv1D_1"))
    best_model.add(BatchNormalization())
    best_model.add(Dropout(0.4))
    best_model.add(Conv1D(filters=112, kernel_size=4, activation="relu", name="Conv1D_2"))
    best_model.add(BatchNormalization())
    best_model.add(Dropout(0.4))
    best_model.add(Conv1D(filters=144, kernel_size=5, activation="relu", name="Conv1D_3"))
    best_model.add(BatchNormalization())
    best_model.add(Dropout(0.0))
    best_model.add(MaxPooling1D(pool_size=2, name="MaxPooling1D"))
    best_model.add(Flatten())
    best_model.add(Dense(units=128, activation="relu", name="Dense_1"))
    best_model.add(Dense(10, name="Dense_2"))
    return best_model


def best_model_SCMneg():
    best_model = Sequential(name="model_conv1D_SCMneg")
    best_model.add(Input(shape=(272, 21)))
    best_model.add(Conv1D(filters=128, kernel_size=5, activation="relu", name="Conv1D_1"))
    best_model.add(BatchNormalization())
    best_model.add(Dropout(0.1))
    best_model.add(Conv1D(filters=112, kernel_size=4, activation="relu", name="Conv1D_2"))
    best_model.add(BatchNormalization())
    best_model.add(Conv1D(filters=64, kernel_size=4, activation="relu", name="Conv1D_3"))
    best_model.add(BatchNormalization())
    best_model.add(MaxPooling1D(pool_size=2, name="MaxPooling1D"))
    best_model.add(Flatten())
    best_model.add(Dense(units=128, activation="relu", name="Dense_1"))
    best_model.add(Dense(10, name="Dense_2"))
    return best_model

def main():
    project_root = get_project_root()
    data_dir = project_root / "data"
    # Create data_dir if it doesn't exist (though it should for input files)
    data_dir.mkdir(parents=True, exist_ok=True)


    # ts = 0.2; bs = 64
    filenames_data = ["Deep_SAPpos_data.txt", "Deep_SCMpos_data.txt", "Deep_SCMneg_data.txt"]
    # Ensure model functions are called to create new instances for each loop iteration if that's the intent
    # Or define them once if they are meant to be the same architecture object reused (Keras might complain)
    # For safety, let's ensure they are callable to reinstantiate
    model_fns = [best_model_SAPpos, best_model_SCMpos, best_model_SCMneg]
    l_rates = [0.0001, 0.005, 0.0001]

    all_results_dfs = []
    all_history_dfs = []

    for data_filename, model_fn, l_rate in zip(filenames_data, model_fns, l_rates):
        prop_type = data_filename.split("_")[1] # e.g., "SAPpos"
        current_model = model_fn() # Instantiate a new model

        print(f"\nProcessing: {data_filename} for property type: {prop_type}")

        name_list, seq_list, score_list = load_input_data_for_training(data_dir / data_filename)
        X = seq_list
        y = score_list

        # Train and compile model with best hyperparameters
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.25, random_state=0 # 0.25 * 0.8 = 0.2, so 60/20/20 split
        )

        X_train_encoded = np.array([one_hot_encoder(s=x) for x in X_train])
        X_train_encoded = np.transpose(X_train_encoded, (0, 2, 1))

        X_test_encoded = np.array([one_hot_encoder(s=x) for x in X_test])
        X_test_encoded = np.transpose(X_test_encoded, (0, 2, 1))

        X_val_encoded = np.array([one_hot_encoder(s=x) for x in X_val])
        X_val_encoded = np.transpose(X_val_encoded, (0, 2, 1))

        y_train_arr = np.asarray(y_train).reshape((-1, 10))
        y_test_arr = np.asarray(y_test).reshape((-1, 10))
        y_val_arr = np.asarray(y_val).reshape((-1, 10))

        optimizer = Adam(learning_rate=l_rate)
        current_model.compile(optimizer=optimizer, loss="mae", metrics=["mae"]) # Added mae metric for history

        # Create callback
        model_weights_path = data_dir / f"Conv1D_regression_{prop_type}.h5"
        model_json_path = data_dir / f"Conv1D_regression{prop_type}.json"

        checkpoint = ModelCheckpoint(
            filepath=str(model_weights_path), # Ensure filepath is string
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=True, # Only save weights, architecture saved separately
            mode="min",
        )
        callbacks_list = [checkpoint]

        # Fit the CNN to the training set
        print(f"Fitting model for {prop_type}...")
        history = current_model.fit(
            x=X_train_encoded,
            y=y_train_arr,
            shuffle=True,
            validation_data=(X_val_encoded, y_val_arr),
            epochs=2, # Temporarily reduced for testing
            callbacks=callbacks_list,
            batch_size=64,
            verbose=2,
        )

        # Save the Conv1D architecture to json
        conv1d_regression_json = current_model.to_json()
        with open(model_json_path, "w") as json_file:
            json_file.write(conv1d_regression_json)
        print(f"Saved model architecture to {model_json_path}")

        # Load the best weights explicitly for prediction (ModelCheckpoint saved them)
        # The current_model object might not have the best weights if training ended on a worse epoch
        # Re-instantiate and load best weights
        pred_model_architecture = model_fn() # Get a fresh model structure
        pred_model_architecture.load_weights(str(model_weights_path)) # Load best weights
        pred_model_architecture.compile(optimizer=optimizer, loss="mae", metrics=["mae"]) # Compile for prediction

        y_pred = pred_model_architecture.predict(X_test_encoded)

        best_val_loss = min(history.history["val_loss"])

        baseline_mae_list = []
        mean_score_list = []
        for i in range(y_test_arr.shape[1]):
            baseline_prediction = np.full_like(y_test_arr[:, i], np.mean(y_train_arr[:, i])) # Use train mean for baseline
            baseline_mae = mean_absolute_error(y_test_arr[:, i], baseline_prediction)
            baseline_mae_list.append(baseline_mae)
            mean_score = np.mean(y_test_arr[:, i]) # Mean of the actual test scores for this target
            mean_score_list.append(mean_score)

        mae_list = []
        corr_list = []
        for i in range(y_test_arr.shape[1]):
            mae = mean_absolute_error(y_test_arr[:, i], y_pred[:, i])
            mae_list.append(mae)
            corr = np.corrcoef(y_test_arr[:, i], y_pred[:, i])[0, 1] if len(y_test_arr[:,i]) > 1 else 0.0 # handle single sample case
            corr_list.append(corr)

        region_names = ["CDRH1", "CDRH2", "CDRH3", "CDRL1", "CDRL2", "CDRL3", "CDR", "Hv", "Lv", "Fv"]
        result_rows = []
        for region_idx, reg_name in enumerate(region_names):
            result_rows.append({
                "prop": f"{prop_type}_{reg_name}", # Corrected property name
                "Mean_score": mean_score_list[region_idx],
                "Baseline_MAE": baseline_mae_list[region_idx],
                "Val_loss": best_val_loss, # This is overall model val_loss, not per-target
                "MAE": mae_list[region_idx],
                "R": corr_list[region_idx],
            })

        result_df = pd.DataFrame(result_rows)
        metric_output_path = data_dir / f"hyp_metric_{prop_type}.csv"
        result_df.to_csv(metric_output_path, index=False)
        print(f"Saved metrics to {metric_output_path}")
        all_results_dfs.append(result_df)

        history_df = pd.DataFrame(history.history)
        history_output_path = data_dir / f"training_history_{prop_type}.csv" # Save in data_dir
        history_df.to_csv(history_output_path, index=False)
        print(f"Saved training history to {history_output_path}")
        all_history_dfs.append(history_df) # Though usually not concatenated

    # Concatenate all metric DataFrames
    if all_results_dfs:
        final_metric_df = pd.concat(all_results_dfs, ignore_index=True)
        final_metric_output_path = data_dir / "Final_model_metric.csv"
        final_metric_df.to_csv(final_metric_output_path, index=False)
        print(f"Saved combined metrics to {final_metric_output_path}")
    else:
        print("No results to concatenate for final metrics.")

if __name__ == "__main__":
    main()

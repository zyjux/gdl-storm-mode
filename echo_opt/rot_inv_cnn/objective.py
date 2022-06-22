from glob import glob
from os.path import join
import xarray as xr
import pandas as pd
import numpy as np
import optuna
from echo.src.base_objective import BaseObjective
from echo.src.pruners import KerasPruningCallback
import sys
sys.path.append("/glade/u/home/lverhoef/gdl-storm-mode/echo_opt/rot_inv_cnn")
from imports.GDL_model import gdl_model


def custom_updates(trial, conf):

    hyperparameters = conf["optuna"]["parameters"]

    filter1 = trial.suggest_int(**hyperparameters["filter1"]["settings"])
    filter2 = trial.suggest_int(**hyperparameters["filter2"]["settings"])
    filter3 = trial.suggest_int(**hyperparameters["filter3"]["settings"])
    filter4 = trial.suggest_int(**hyperparameters["filter4"]["settings"])
    kernel1 = trial.suggest_int(**hyperparameters["kernel1"]["settings"])
    kernel2 = trial.suggest_int(**hyperparameters["kernel2"]["settings"])
    kernel3 = trial.suggest_int(**hyperparameters["kernel3"]["settings"])
    kernel4 = trial.suggest_int(**hyperparameters["kernel4"]["settings"])
    pool1 = trial.suggest_int(**hyperparameters["pool1"]["settings"])
    pool2 = trial.suggest_int(**hyperparameters["pool2"]["settings"])
    pool3 = trial.suggest_int(**hyperparameters["pool3"]["settings"])

    conf["model"]["filters"] = [filter1, filter2, filter3, filter4]
    conf["model"]["kernel_sizes"] = [kernel1, kernel2, kernel3, kernel4]
    conf["model"]["pool_sizes"] = [pool1, pool2, pool3, 1]

    return conf


class Objective(BaseObjective):
    def __init__(self, config, metric='val_loss'):
        BaseObjective.__init__(self, config, metric)

    def train(self, trial, conf):

        # Make custom updates to model conf
        conf = custom_updates(trial, conf)

        # Find a list of all the datafiles
        patch_path = "/glade/scratch/lverhoef/WRF_all/track_data_hrrr_3km_nc_refl/"
        patch_files = sorted(glob(join(patch_path, "*.nc")))
        csv_path = "/glade/scratch/lverhoef/WRF_all/track_data_hrrr_3km_csv_refl/"
        csv_files = sorted(glob(join(csv_path, "track_step_*.csv")))

        # Pull selected variables from patch files and join into a single DataSet
        num_files = 100
        variables = ["i", "j", "REFL_COM_curr"]
        data_list = []
        for p, patch_file in enumerate(patch_files[0:num_files]):
            ds = xr.open_dataset(patch_file)
            data_list.append(ds[variables].compute())
            ds.close()
        data = xr.concat(data_list, dim="p")

        # Pull variables from csv files and join into an array
        csv_variables = ["major_axis_length", "minor_axis_length"]
        csv_data_list = []
        for p, csv_file in enumerate(csv_files[0:num_files]):
            csv_ds = pd.read_csv(csv_file)
            csv_data_list.append(csv_ds[csv_variables].to_xarray().rename({'index': 'p'}))
        csv_data = xr.concat(csv_data_list, dim="p")

        # Create DataArrays for input and output data
        input_data = data["REFL_COM_curr"].expand_dims("channel", axis=-1)
        output_data = csv_data.to_array().transpose()

        # Find indices to split data into 70% training, 10% validation, and 20% test. The training and validation data are shuffled, while the test data is temporally different.
        # rng = np.random.default_rng()
        split_point_1 = int(0.7 * input_data.shape[0])
        split_point_2 = int(0.8 * input_data.shape[0])
        train_val_indices = np.arange(0, split_point_2)
        # rng.shuffle(train_val_indices)
        train_indices = train_val_indices[:split_point_1]
        val_indices = train_val_indices[split_point_1:]

        # Normalize the training input data and actually evaluate the input_train array which will be fed into the network
        input_train = input_data.values[train_indices]
        scale_stats = pd.DataFrame(index=[0], columns=["mean", "sd"])
        scale_stats.loc[0, "mean"] = input_train.mean()
        scale_stats.loc[0, "sd"] = input_train.std()
        input_train_norm = (input_train - scale_stats.loc[0, "mean"]) / scale_stats.loc[0, "sd"]

        # Normalize the validation data
        input_val = input_data.values[val_indices]
        input_val_norm = (input_val - scale_stats.loc[0, "mean"]) / scale_stats.loc[0, "sd"]

        # Split output into train, test, and validation sets
        output_train = output_data[train_indices]
        output_val = output_data[val_indices]

        model = gdl_model(**conf["model"])

        callbacks = [KerasPruningCallback(trial, self.metric, interval=1)]
        try:
            result = model.fit(input_train_norm, output_train, xv=input_val_norm, yv=output_val, callbacks=callbacks)
        except ValueError:
            raise optuna.TrialPruned()

        results_dictionary = {
            "train_loss": result["loss"],
            "val_loss": result["val_loss"],
            "train_mse": result["mse"],
            "val_mse": result["val_mse"]
        }

        return results_dictionary
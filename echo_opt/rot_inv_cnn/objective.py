from glob import glob
from os.path import join
import xarray as xr
import pandas as pd
import numpy as np
import optuna
from echo.src.base_objective import BaseObjective
from keras import backend as K
import gc
import sys
from timeit import default_timer as timer
sys.path.append("/glade/u/home/lverhoef/gdl-storm-mode/echo_opt/rot_inv_cnn")
from imports.GDL_model import gdl_model


def custom_updates(trial, conf):

    hyperparameters = conf["optuna"]["parameters"]

    filter1 = trial.suggest_int(**hyperparameters["filter1"]["settings"])
    filter2 = trial.suggest_int(**hyperparameters["filter2"]["settings"])
    filter3 = trial.suggest_int(**hyperparameters["filter3"]["settings"])
    filter4 = trial.suggest_int(**hyperparameters["filter4"]["settings"])
    kernel1 = trial.suggest_categorical(**hyperparameters["kernel1"]["settings"])
    kernel2 = trial.suggest_categorical(**hyperparameters["kernel2"]["settings"])
    kernel3 = trial.suggest_categorical(**hyperparameters["kernel3"]["settings"])
    kernel4 = trial.suggest_categorical(**hyperparameters["kernel4"]["settings"])
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
        # conf = custom_updates(trial, conf)

        # Find a list of all the datafiles
        patch_path = "/glade/scratch/lverhoef/WRF_all/track_data_hrrr_3km_nc_refl/"
        patch_files = sorted(glob(join(patch_path, "*.nc")))
        csv_path = "/glade/scratch/lverhoef/WRF_all/track_data_hrrr_3km_csv_refl/"
        csv_files = sorted(glob(join(csv_path, "track_step_*.csv")))

        # Pull selected variables from patch files and join into Datasets
        num_files = 150
        train_split = int(num_files*0.7)
        val_split = int(num_files*0.8)
        variables = ["REFL_COM_curr"]
        data_list = []
        for p, patch_file in enumerate(patch_files[0:train_split]):
            if p % 10 == 0:
                print(f'Train {p}, {patch_file}')
            ds = xr.open_dataset(patch_file)
            data_list.append(ds[variables].compute())
            ds.close()
        input_train = xr.concat(data_list, dim="p")["REFL_COM_curr"].expand_dims("channel", axis = -1)
        data_list = []
        for p, patch_file in enumerate(patch_files[train_split:val_split]):
            if p % 10 == 0:
                print(f'Validation {train_split + p}, {patch_file}')
            ds = xr.open_dataset(patch_file)
            data_list.append(ds[variables].compute())
            ds.close()
        input_val = xr.concat(data_list, dim="p")["REFL_COM_curr"].expand_dims("channel", axis = -1)
        data_list = []
        for p, patch_file in enumerate(patch_files[val_split:num_files]):
            if p % 10 == 0:
                print(f'Test {val_split + p}, {patch_file}')
            ds = xr.open_dataset(patch_file)
            data_list.append(ds[variables].compute())
            ds.close()
        input_test = xr.concat(data_list, dim="p")["REFL_COM_curr"].expand_dims("channel", axis = -1)

        # Pull variables from csv files and join into an array
        
        # Pull variables from csv files and join into an array
        csv_variables = ["major_axis_length", "minor_axis_length"]
        csv_data_list = []
        for p, csv_file in enumerate(csv_files[0:train_split]):
            if p % 10 == 0:
                print(f'Train {p}, {csv_file}')
            csv_ds = pd.read_csv(csv_file)
            csv_data_list.append(csv_ds[csv_variables].to_xarray().rename({'index': 'p'}))
        output_train = xr.concat(csv_data_list, dim="p").to_array().transpose()
        csv_data_list = []
        for p, csv_file in enumerate(csv_files[train_split:val_split]):
            if p % 10 == 0:
                print(f'Validation {train_split + p}, {csv_file}')
            csv_ds = pd.read_csv(csv_file)
            csv_data_list.append(csv_ds[csv_variables].to_xarray().rename({'index': 'p'}))
        output_val = xr.concat(csv_data_list, dim="p").to_array().transpose()
        csv_data_list = []
        for p, csv_file in enumerate(csv_files[val_split:num_files]):
            if p % 10 == 0:
                print(f'Test {val_split + p}, {csv_file}')
            csv_ds = pd.read_csv(csv_file)
            csv_data_list.append(csv_ds[csv_variables].to_xarray().rename({'index': 'p'}))
        output_test = xr.concat(csv_data_list, dim="p").to_array().transpose()

        # Normalize the input data
        scale_stats = pd.DataFrame(index=[0], columns=["mean", "sd"])
        scale_stats.loc[0, "mean"] = input_train.mean()
        scale_stats.loc[0, "sd"] = input_train.std()
        input_train_norm = (input_train - scale_stats.loc[0, "mean"]) / scale_stats.loc[0, "sd"]
        input_val_norm = (input_val - scale_stats.loc[0, "mean"]) / scale_stats.loc[0, "sd"]
        input_test_norm = (input_test - scale_stats.loc[0, "mean"]) / scale_stats.loc[0, "sd"]
        
        # Normalize the output data
        output_scale_stats = pd.DataFrame(index=range(output_train.shape[-1]), columns=["mean", "sd"])
        output_scale_stats.loc[:, "mean"] = output_train.mean(dim="p")
        output_scale_stats.loc[:, "sd"] = output_train.std(dim="p")
        output_train_norm = xr.DataArray(coords=output_train.coords, dims=output_train.dims)
        output_val_norm = xr.DataArray(coords=output_val.coords, dims=output_val.dims)
        output_test_norm = xr.DataArray(coords=output_test.coords, dims=output_test.dims)
        for i in range(output_train.shape[-1]):
            output_train_norm[:, i] = (output_train[:, i] - output_scale_stats.loc[i, "mean"]) / output_scale_stats.loc[i, "sd"]
            output_val_norm[:, i] = (output_val[:, i] - output_scale_stats.loc[i, "mean"]) / output_scale_stats.loc[i, "sd"]
            output_test_norm[:, i] = (output_test[:, i] - output_scale_stats.loc[i, "mean"]) / output_scale_stats.loc[i, "sd"]

        model = gdl_model(**conf["model"])

        callbacks = []  # [KerasPruningCallback(trial, self.metric, interval=1)]
        try:
            start = timer()
            result = model.fit(input_train_norm, output_train_norm, xv=input_val_norm, yv=output_val_norm, callbacks=callbacks)
            tot_time = timer() - start
        except:
            raise optuna.TrialPruned()

        results_dictionary = {
            "train_loss": result["loss"],
            "val_loss": result["val_loss"],
            "val_loss_best": min(result["val_loss"]),
            "tot_time": tot_time
        }

        K.clear_session()
        del model
        gc.collect()

        return results_dictionary
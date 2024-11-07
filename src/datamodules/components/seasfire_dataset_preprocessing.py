import xbatcher
import xarray as xr
from tqdm import tqdm
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from torchvision import transforms
import random
import time
import os
import json
import numpy as np


def sample_dataset(ds, input_vars, target, target_shift, output_path, split='train', dim_lon=128, dim_lat=128, dim_time=2, num_timesteps=-1, batch_size=50):
    print(f'Shifting inputs by {-target_shift}')
    
    # Shift target by time
    for var in input_vars:
        if target_shift < 0:
            ds[var] = ds[var].shift(time=-target_shift)

    # Slice dataset based on split type
    if split == 'train':
        ds = ds.sel(time=slice('2002-01-01', '2018-01-01'))
        output_dir = output_path +"/train"
    elif split == 'val':
        ds = ds.sel(time=slice('2018-01-01', '2019-01-01'))
        output_dir = output_path + "/val"
    elif split == 'test':
        ds = ds.sel(time=slice('2019-01-01', '2020-01-01'))
        output_dir = output_path + "/test"

    # Limit timesteps if specified
    if num_timesteps > 0:
        ds = ds.isel(time=slice(0, num_timesteps-1))

    # Select only the necessary variables
    ds = ds[input_vars + [target]]

    # Chunk data along the time axis for lazy evaluation
    ds = ds.chunk({'time': batch_size})
    print("Dataset ready for batching")

    # Initialize mean and std dictionaries for normalization
    mean_std_dict = {}
    for var in input_vars + [target]:
        mean_std_dict[var + '_mean'] = ds[var].mean().values.item(0)
        mean_std_dict[var + '_std'] = ds[var].std().values.item(0)    # Lazy evaluation

    # Save the mean_std_dict to a JSON file
    means_path = output_dir + "/mean_std.json"
    os.makedirs(os.path.dirname(means_path), exist_ok=True)
    with open(means_path, 'w') as f:
        json.dump(mean_std_dict, f)
    print(f"Mean and standard deviations saved to {means_path}")


    # Create directory to save positive batches if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over batches and save positives to disk
    n_batches = 0
    n_pos = 0
    saved_batches_info = []  # Metadata about saved batches
    num_positive_pixels = 0
    num_null_pixels = 0
    for i in range(0, ds.sizes['time'], batch_size):
        batch_ds = ds.isel(time=slice(i, i + batch_size))

        # Check if positive values exist before loading
        if batch_ds[target].sum().compute() > 0:
            batch_ds = batch_ds.compute()  # Load into memory only when necessary
            bgen = xbatcher.BatchGenerator(
                ds=batch_ds,
                input_dims={'longitude': dim_lon, 'latitude': dim_lat, 'time': dim_time},
                input_overlap={'time': dim_time - 1} if (dim_time - 1) else {}
            )
            for j, batch in enumerate(bgen):
                if batch.isel(time=-1)[target].sum().compute() > 0:
                    target_data = batch.isel(time=-1)[target].copy()
                    target_data = target_data.compute()
                    target_data = np.nan_to_num(target_data, nan=0)
                    target_data = np.where(target_data != 0, 1, 0)
                    num_positive_pixels +=  (target_data > 0).sum()
                    num_null_pixels += (target_data == 0).sum()
                    # Save batch with positive target values to disk
                    batch_filename = f"{output_dir}/batch_{i}_{j}.nc"
                    batch.to_netcdf(batch_filename)

                    # Keep track of the saved batch info (file path and indices)
                    saved_batches_info.append({
                        'file_path': batch_filename,
                        'batch_index': (i, j)
                    })
                    n_pos += 1
            n_batches += 1

    print('# of batches', n_batches)
    print('# of positives', n_pos)
    print('# of positive pixels', num_positive_pixels)
    print('# of null pixels', num_null_pixels)
    # Specify the filename
    batch_path = output_dir +'/saved_batches_info.json'
    os.makedirs(os.path.dirname(batch_path), exist_ok=True)


    # Save to JSON
    with open(batch_path, 'w') as f:
        json.dump(saved_batches_info, f)

    return batch_path, means_path, n_pos


class BatcherDS(Dataset):
    def __init__(self, batches_path, input_vars, positional_vars, target, mean_std_dict,task='classification'):
        """
        batches: List of dictionaries containing metadata of each batch (e.g., file paths)
        input_vars: List of input variable names
        positional_vars: List of positional variable names (e.g., latitude, longitude)
        target: The name of the target variable
        mean_std_dict: Dictionary with mean and std values for normalization
        """

        # Load the JSON file
        with open(batches_path, 'r') as json_file:
            batches = json.load(json_file)
        with open(mean_std_dict, 'r') as json_file:
            mean_std_dict = json.load(json_file)

        self.task = task
        self.batches = batches
        self.input_vars = input_vars
        self.positional_vars = positional_vars
        self.target = target
        self.mean_std_dict = mean_std_dict

        self.mean = np.stack([mean_std_dict[f'{var}_mean'] for var in input_vars])
        self.std = np.stack([mean_std_dict[f'{var}_std'] for var in input_vars])

    def __len__(self):
        # Return the total number of batches
        return len(self.batches)

    def __getitem__(self, idx):
        # Get the batch metadata (e.g., file path)
        batch_info = self.batches[idx]
        batch_file = batch_info['file_path']

        # Load the batch from disk
        batch_ds = xr.open_dataset(batch_file)

        # Extract the input and target variables
        inputs = [batch_ds[var].values for var in self.input_vars]

        #positional = [batch_ds[var].values for var in self.positional_vars]
        target = batch_ds[self.target].values

        # Normalize inputs using the provided mean and std
        for i, var in enumerate(self.input_vars):
            inputs[i] = (inputs[i] - self.mean_std_dict[f'{var}_mean']) / self.mean_std_dict[f'{var}_std']

        inputs = [x.reshape(128, 128) for x in inputs]
        target = target[0]

        inputs = np.nan_to_num(inputs, nan=-1)
        target = np.nan_to_num(target, nan=0)
        if self.task == 'classification':
             target = np.where(target != 0, 1, 0)

        return  inputs, target
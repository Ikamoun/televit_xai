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
# def sample_dataset(ds, input_vars, target, target_shift, split='train', dim_lon=128, dim_lat=128, dim_time=2, num_timesteps=-1, hundreds = 0):
#     print(f'Shifting inputs by {-target_shift}')
#     for var in input_vars:
#         if target_shift < 0:
#             ds[var] = ds[var].shift(time=-target_shift)

#     # if split == 'train':
#     #     ds = ds.sel(time=slice('2002-01-01', '2018-01-01'))
#     # elif split == 'val':
#     #     ds = ds.sel(time=slice('2018-01-01', '2019-01-01'))
#     # elif split == 'test':
#     #     ds = ds.sel(time=slice('2019-01-01', '2020-01-01'))

#     if split == 'train':
#         ds = ds.sel(time=slice('2002-01-01', '2018-01-01'))
#         save_path = '/data/ines/train_set_' +str(hundreds)
#     elif split == 'val':
#         ds = ds.sel(time=slice('2018-01-01', '2019-01-01'))
#         save_path = '/data/ines/val_set'
#     elif split == 'test':
#         ds = ds.sel(time=slice('2019-01-01', '2020-01-01'))
#         save_path = '/data/ines/test_set'

#     if num_timesteps > 0 and num_timesteps < 100:
#         # If num_timesteps is between 1 and 99, slice the dataset
#         ds = ds.isel(time=slice(0, num_timesteps))  # Note: -1 is not needed, the upper limit is exclusive
#     elif (num_timesteps >= 100 and split == 'train') or (num_timesteps < 0 and split == 'train'):
#         # Calculate how many complete intervals of 100 fit into num_timesteps
#         hundred = num_timesteps // 100
#         for time_interval in range(hundred):
#             # Slice the dataset for each 100 time step interval
#             start = 100 * time_interval
#             end = 100 * (time_interval + 1)  # This will give you the exclusive upper limit
#             chunk = ds.isel(time=slice(start, end))
#             # Process the chunk with sample_dataset_load
#             sample_dataset(chunk, input_vars, target, target_shift, split='train', dim_lon=128, dim_lat=128, dim_time=2, num_timesteps=-1, hundreds = time_interval)
#         # Process any remaining time steps after the complete intervals
#         remaining_start = 100 * hundred
#         if remaining_start < num_timesteps:
#             chunk = ds.isel(time=slice(remaining_start, num_timesteps))  # Get remaining time steps
#             sample_dataset(chunk, input_vars, target, target_shift, split='train', dim_lon=128, dim_lat=128, dim_time=2, num_timesteps=-1, hundreds = time_interval + 1 )

#     ds = ds[input_vars + [target]]
#     ds = ds.load()
#     print("Dataset loaded")
#     bgen = xbatcher.BatchGenerator(
#         ds=ds,
#         input_dims={'longitude': dim_lon, 'latitude': dim_lat, 'time': dim_time},
#         input_overlap={'time': dim_time - 1} if (dim_time - 1) else {}
#     )


#     # compute positional embedding from longitude and latitude
#     lon = ds.longitude.values
#     lat = ds.latitude.values
#     lon = np.expand_dims(lon, axis=0)
#     lat = np.expand_dims(lat, axis=1)
#     lon = np.tile(lon, (lat.shape[0], 1))
#     lat = np.tile(lat, (1, lon.shape[1]))

#     ds['cos_lon'] = ({'latitude': ds.latitude, 'longitude': ds.longitude}, np.cos(lon * np.pi / 180))
#     ds['cos_lat'] = ({'latitude': ds.latitude, 'longitude': ds.longitude}, np.cos(lat * np.pi / 180))
#     ds['sin_lon'] = ({'latitude': ds.latitude, 'longitude': ds.longitude}, np.sin(lon * np.pi / 180))
#     ds['sin_lat'] = ({'latitude': ds.latitude, 'longitude': ds.longitude}, np.sin(lat * np.pi / 180))
#     # if log_tp:
#     ds['tp'] = np.log(ds['tp'] + 1)
#     ds[target] = np.log(ds[target] + 1)


#     means_path = os.path.join(save_path, 'mean_std_values.npy') 
#     # calclulate sum of gwis_ba in a rolling window of 1 month


#     n_batches = 0
#     n_pos = 0
#     positives = []
#     negatives = []
#     batches = []
#     mean_std_dict = {}
#     for var in input_vars + [target]:
#         mean_std_dict[var + '_mean'] = ds[var].mean().values.item(0)
#         mean_std_dict[var + '_std'] = ds[var].std().values.item(0)
#     for batch in tqdm(bgen):
#         if batch.isel(time=-1)[target].sum() > 0:


#             positives.append(batch)
#             n_pos += 1
#         #         else:
#         #             if not np.isnan(batch.isel(time=-1)['NDVI']).sum()>0:
#         #                 negatives.append(batch)
#         n_batches += 1
#     print('# of batches', n_batches)
#     print('# of positives', n_pos)

#     os.makedirs(os.path.dirname(means_path), exist_ok=True)
#     np.save(means_path, mean_std_dict)
#     print('data saved')

#     positives_concat = xr.concat(positives, dim='time')
#     positives_path = os.path.join(save_path, 'positives.nc')
#     positives_concat.to_netcdf(positives_path)  # Save to NetCDF format

#     return positives, mean_std_dict, n_pos


# def sample_dataset(ds, input_vars, target, target_shift, split='train', dim_lon=128, dim_lat=128, dim_time=2, num_timesteps=-1, batch_size=50):
#     """
#     Sample dataset in batches instead of loading the whole dataset at once.
#     Args:
#         ds: Xarray Dataset to process.
#         input_vars: List of input variable names.
#         target: Target variable name.
#         target_shift: Time shift to apply to inputs.
#         split: Data split ('train', 'val', or 'test').
#         dim_lon: Longitude dimension size.
#         dim_lat: Latitude dimension size.
#         dim_time: Time dimension size.
#         num_timesteps: Maximum number of timesteps to use from the dataset.
#         batch_size: Number of time steps to process in each batch (default 100).
#     """
#     print(f'Shifting inputs by {-target_shift}')
#     for var in input_vars:
#         if target_shift < 0:
#             ds[var] = ds[var].shift(time=-target_shift)

#     if split == 'train':
#         ds = ds.sel(time=slice('2004-01-01', '2018-01-01'))
#     elif split == 'val':
#         ds = ds.sel(time=slice('2018-01-01', '2019-01-01'))
#     elif split == 'test':
#         ds = ds.sel(time=slice('2019-01-01', '2020-01-01'))

#     if num_timesteps > 0:
#         ds = ds.isel(time=slice(0, num_timesteps))

#     ds = ds[input_vars + [target]]

#     print("Dataset ready for batching")

#     # Initialize mean and std dictionaries for normalization
#     mean_std_dict = {}
#     for var in input_vars + [target]:
#         mean_std_dict[var + '_mean'] = ds[var].mean().values.item(0)
#         mean_std_dict[var + '_std'] = ds[var].std().values.item(0)

#     # Slice the dataset into batches of 'batch_size' timesteps
#     n_timesteps = ds.sizes['time']
#     positives = []
#     n_batches = 0
#     n_pos = 0

#     for i in range(0, n_timesteps, batch_size):
#         # Slice dataset for the current batch
#         batch_ds = ds.isel(time=slice(i, i + batch_size))
#         batch_ds = batch_ds.load()
#         print(f"Processing batch from timestep {i} to {i + batch_size}")

#         # Create batches from the sliced dataset
#         bgen = xbatcher.BatchGenerator(
#             ds=batch_ds,
#             input_dims={'longitude': dim_lon, 'latitude': dim_lat, 'time': dim_time},
#             input_overlap={'time': dim_time - 1} if (dim_time - 1) else {}
#         )

#         # Iterate through the generated batches and process them
#         for batch in tqdm(bgen):
#             # Check if the last time slice in this batch has a positive target value
#             if batch.isel(time=-1)[target].sum() > 0:
#                 positives.append(batch)
#                 n_pos += 1
#             n_batches += 1

#     print('# of batches', n_batches)
#     print('# of positives', n_pos)

#     return positives, mean_std_dict, n_pos




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


# class BatcherDS(Dataset):
#     """Dataset from Xbatcher"""

#     def __init__(self, batches, input_vars, positional_vars, target, mean_std_dict, task='classification'):
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.task = task
#         self.batches = batches
#         self.target = target
#         self.input_vars = input_vars
#         self.mean_std_dict = mean_std_dict
#         self.positional_vars = positional_vars
#         self.mean = np.stack([mean_std_dict[f'{var}_mean'] for var in input_vars])
#         self.std = np.stack([mean_std_dict[f'{var}_std'] for var in input_vars])

#     def __len__(self):
#         return len(self.batches)

#     def __getitem__(self, idx):
#         batch = self.batches[idx].isel(time=-1)
#         inputs = np.stack([batch[var] for var in self.input_vars + self.positional_vars]).astype(np.float32)
#         for i, var in enumerate(self.input_vars):
#             inputs[i] = (inputs[i] - self.mean_std_dict[f'{var}_mean']) / self.mean_std_dict[f'{var}_std']
#         target = batch[self.target].values
#         inputs = np.nan_to_num(inputs, nan=-1)
#         target = np.nan_to_num(target, nan=0)
#         # make this a classification dataset
#         if self.task == 'classification':
#             target = np.where(target != 0, 1, 0)
#         return inputs, target


class BatcherDS(Dataset):
    def __init__(self, batches_path, input_vars, positional_vars, target, mean_std_dict,task='classification', push_prototypes = False):
        """
        batches: List of dictionaries containing metadata of each batch (e.g., file paths)
        input_vars: List of input variable names
        positional_vars: List of positional variable names (e.g., latitude, longitude)
        target: The name of the target variable
        mean_std_dict: Dictionary with mean and std values for normalization
        push_prototypes: to normalize or not #to do changge this 
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
        self.push_prototypes = push_prototypes

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

        #if not self.push_prototypes:
        # Normalize inputs using the provided mean and std
        for i, var in enumerate(self.input_vars):
            inputs[i] = (inputs[i] - self.mean_std_dict[f'{var}_mean']) / self.mean_std_dict[f'{var}_std']

        inputs = [x.reshape(128, 128) for x in inputs]
        target = target[0]
        # Convert inputs and target to PyTorch tensors
        #inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        #positional_tensor = torch.tensor(positional, dtype=torch.float32)
        #target_tensor = torch.tensor(target, dtype=torch.float32)
        
        inputs = np.nan_to_num(inputs, nan=-1)
        target = np.nan_to_num(target, nan=0)
        if self.task == 'classification':
             target = np.where(target != 0, 1, 0)

        return  inputs, target
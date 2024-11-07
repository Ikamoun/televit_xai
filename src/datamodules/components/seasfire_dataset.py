import xbatcher
import xarray as xr
from tqdm import tqdm
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from torchvision import transforms
import random
import time
import numpy as np


def sample_dataset(ds, input_vars, target, target_shift, split='train', dim_lon=128, dim_lat=128, dim_time=2, num_timesteps=-1):
    print(f'Shifting inputs by {-target_shift}')
    for var in input_vars:
        if target_shift < 0:
            ds[var] = ds[var].shift(time=-target_shift)

    # if split == 'train':
    #     ds = ds.sel(time=slice('2002-01-01', '2018-01-01'))
    # elif split == 'val':
    #     ds = ds.sel(time=slice('2018-01-01', '2019-01-01'))
    # elif split == 'test':
    #     ds = ds.sel(time=slice('2019-01-01', '2020-01-01'))

    if split == 'train':
        ds = ds.sel(time=slice('2002-01-01', '2018-01-01'))
    elif split == 'val':
        ds = ds.sel(time=slice('2018-01-01', '2019-01-01'))
    elif split == 'test':
        ds = ds.sel(time=slice('2019-01-01', '2020-01-01'))

    if num_timesteps > 0:
        ds = ds.isel(time=slice(0, num_timesteps - 1))
    

    ds = ds[input_vars + [target]]
    ds = ds.load()
    print("Dataset loaded")
    bgen = xbatcher.BatchGenerator(
        ds=ds,
        input_dims={'longitude': dim_lon, 'latitude': dim_lat, 'time': dim_time},
        input_overlap={'time': dim_time - 1} if (dim_time - 1) else {}
    )


    # compute positional embedding from longitude and latitude
    lon = ds.longitude.values
    lat = ds.latitude.values
    lon = np.expand_dims(lon, axis=0)
    lat = np.expand_dims(lat, axis=1)
    lon = np.tile(lon, (lat.shape[0], 1))
    lat = np.tile(lat, (1, lon.shape[1]))

    ds['cos_lon'] = ({'latitude': ds.latitude, 'longitude': ds.longitude}, np.cos(lon * np.pi / 180))
    ds['cos_lat'] = ({'latitude': ds.latitude, 'longitude': ds.longitude}, np.cos(lat * np.pi / 180))
    ds['sin_lon'] = ({'latitude': ds.latitude, 'longitude': ds.longitude}, np.sin(lon * np.pi / 180))
    ds['sin_lat'] = ({'latitude': ds.latitude, 'longitude': ds.longitude}, np.sin(lat * np.pi / 180))
    # if log_tp:
    ds['tp'] = np.log(ds['tp'] + 1)
    ds[target] = np.log(ds[target] + 1)


    # calclulate sum of gwis_ba in a rolling window of 1 month


    n_batches = 0
    n_pos = 0
    positives = []
    negatives = []
    batches = []
    mean_std_dict = {}
    for var in input_vars + [target]:
        mean_std_dict[var + '_mean'] = ds[var].mean().values.item(0)
        mean_std_dict[var + '_std'] = ds[var].std().values.item(0)
    for batch in tqdm(bgen):
        if batch.isel(time=-1)[target].sum() > 0:


            positives.append(batch)
            n_pos += 1
        #         else:
        #             if not np.isnan(batch.isel(time=-1)['NDVI']).sum()>0:
        #                 negatives.append(batch)
        n_batches += 1
    print('# of batches', n_batches)
    print('# of positives', n_pos)
    return positives, mean_std_dict, n_pos


    # def sample_dataset_load(ds, input_vars, target, target_shift, split='train', dim_lon=128, dim_lat=128, dim_time=2, num_timesteps=-1, hundreds = 0):
    # print(f'Shifting inputs by {-target_shift}')
    # for var in input_vars:
    #     if target_shift < 0:
    #         ds[var] = ds[var].shift(time=-target_shift)

    # # if split == 'train':
    # #     ds = ds.sel(time=slice('2002-01-01', '2018-01-01'))
    # # elif split == 'val':
    # #     ds = ds.sel(time=slice('2018-01-01', '2019-01-01'))
    # # elif split == 'test':
    # #     ds = ds.sel(time=slice('2019-01-01', '2020-01-01'))

    # if split == 'train':
    #     ds = ds.sel(time=slice('2002-01-01', '2018-01-01'))
    #     save_path = '/data/ines/train_set_' +str(hundreds)
    # elif split == 'val':
    #     ds = ds.sel(time=slice('2018-01-01', '2019-01-01'))
    #     save_path = '/data/ines/val_set'
    # elif split == 'test':
    #     ds = ds.sel(time=slice('2019-01-01', '2020-01-01'))
    #     save_path = '/data/ines/test_set'

    # if num_timesteps > 0 and num_timesteps < 100:
    #     # If num_timesteps is between 1 and 99, slice the dataset
    #     ds = ds.isel(time=slice(0, num_timesteps))  # Note: -1 is not needed, the upper limit is exclusive
    # elif (num_timesteps >= 100 and split == 'train') or (num_timesteps < 0 and split == 'train'):
    #     # Calculate how many complete intervals of 100 fit into num_timesteps
    #     hundred = num_timesteps // 100
    #     for time_interval in range(hundred):
    #         # Slice the dataset for each 100 time step interval
    #         start = 100 * time_interval
    #         end = 100 * (time_interval + 1)  # This will give you the exclusive upper limit
    #         chunk = ds.isel(time=slice(start, end))
    #         # Process the chunk with sample_dataset_load
    #         sample_dataset_load(chunk, input_vars, target, target_shift, split='train', dim_lon=128, dim_lat=128, dim_time=2, num_timesteps=-1, hundreds = time_interval)
    #     # Process any remaining time steps after the complete intervals
    #     remaining_start = 100 * hundred
    #     if remaining_start < num_timesteps:
    #         chunk = ds.isel(time=slice(remaining_start, num_timesteps))  # Get remaining time steps
    #         sample_dataset_load(chunk, input_vars, target, target_shift, split='train', dim_lon=128, dim_lat=128, dim_time=2, num_timesteps=-1, hundreds = time_interval + 1 )

    # ds = ds[input_vars + [target]]
    # ds = ds.load()
    # print("Dataset loaded")
    # bgen = xbatcher.BatchGenerator(
    #     ds=ds,
    #     input_dims={'longitude': dim_lon, 'latitude': dim_lat, 'time': dim_time},
    #     input_overlap={'time': dim_time - 1} if (dim_time - 1) else {}
    # )


    # # compute positional embedding from longitude and latitude
    # lon = ds.longitude.values
    # lat = ds.latitude.values
    # lon = np.expand_dims(lon, axis=0)
    # lat = np.expand_dims(lat, axis=1)
    # lon = np.tile(lon, (lat.shape[0], 1))
    # lat = np.tile(lat, (1, lon.shape[1]))

    # ds['cos_lon'] = ({'latitude': ds.latitude, 'longitude': ds.longitude}, np.cos(lon * np.pi / 180))
    # ds['cos_lat'] = ({'latitude': ds.latitude, 'longitude': ds.longitude}, np.cos(lat * np.pi / 180))
    # ds['sin_lon'] = ({'latitude': ds.latitude, 'longitude': ds.longitude}, np.sin(lon * np.pi / 180))
    # ds['sin_lat'] = ({'latitude': ds.latitude, 'longitude': ds.longitude}, np.sin(lat * np.pi / 180))
    # # if log_tp:
    # ds['tp'] = np.log(ds['tp'] + 1)
    # ds[target] = np.log(ds[target] + 1)


    # means_path = os.path.join(save_path, 'mean_std_values.npy') 
    # # calclulate sum of gwis_ba in a rolling window of 1 month


    # n_batches = 0
    # n_pos = 0
    # positives = []
    # negatives = []
    # batches = []
    # mean_std_dict = {}
    # for var in input_vars + [target]:
    #     mean_std_dict[var + '_mean'] = ds[var].mean().values.item(0)
    #     mean_std_dict[var + '_std'] = ds[var].std().values.item(0)
    # for batch in tqdm(bgen):
    #     if batch.isel(time=-1)[target].sum() > 0:


    #         positives.append(batch)
    #         n_pos += 1
    #     #         else:
    #     #             if not np.isnan(batch.isel(time=-1)['NDVI']).sum()>0:
    #     #                 negatives.append(batch)
    #     n_batches += 1
    # print('# of batches', n_batches)
    # print('# of positives', n_pos)
    # np.save(means_path, mean_std_dict)

    # positives_concat = xr.concat(positives, dim='time')
    # positives_path = os.path.join(save_path, 'positives.nc')
    # positives_concat.to_netcdf(positives_path)  # Save to NetCDF format

    # return positives, mean_std_dict, n_pos


class BatcherDS(Dataset):
    """Dataset from Xbatcher"""

    def __init__(self, batches, input_vars, positional_vars, target, mean_std_dict, task='classification'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.task = task
        self.batches = batches
        self.target = target
        self.input_vars = input_vars
        self.mean_std_dict = mean_std_dict
        self.positional_vars = positional_vars
        self.mean = np.stack([mean_std_dict[f'{var}_mean'] for var in input_vars])
        self.std = np.stack([mean_std_dict[f'{var}_std'] for var in input_vars])

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        batch = self.batches[idx].isel(time=-1)
        inputs = np.stack([batch[var] for var in self.input_vars + self.positional_vars]).astype(np.float32)
        for i, var in enumerate(self.input_vars):
            inputs[i] = (inputs[i] - self.mean_std_dict[f'{var}_mean']) / self.mean_std_dict[f'{var}_std']
        target = batch[self.target].values
        inputs = np.nan_to_num(inputs, nan=-1)
        target = np.nan_to_num(target, nan=0)
        # make this a classification dataset
        if self.task == 'classification':
            target = np.where(target != 0, 1, 0)
        return inputs, target


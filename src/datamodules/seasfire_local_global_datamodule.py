from typing import Optional, Tuple
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import numpy as np
import xarray as xr
import xbatcher
import json
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from .components.seasfire_dataset_preprocessing import BatcherDS, sample_dataset
import os
from pathlib import Path

class SeasFireLocalGlobalDataModule(LightningDataModule):

    def __init__(
            self,
            ds_path: str = None,
            ds_path_global: str = None,
            output_path: str = None,
            input_vars: list = None,
            positional_vars: list = None,
            log_transform_vars: list = None,
            target: str = 'BurntArea',
            target_shift: int = 1,
            patch_size: list = None,
            batch_size: int = 64,
            num_workers: int = 8,
            pin_memory: bool = False,
            debug: bool = False,
            stats_dir: str = os.getcwd() + '/stats',
    ):
        super().__init__()

        if patch_size is None:
            #patch_size = [1, 80, 80]
            patch_size = [1, 128, 128]
        if positional_vars is None:
            self.positional_vars = []
        else:
            self.positional_vars = positional_vars

        self.save_hyperparameters(logger=False)
        self.ds_path = ds_path
        self.output_path = output_path
        self.ds_path_global = ds_path_global
        self.input_vars = list(input_vars)
        self.target = target
        self.target_shift = target_shift
        self.ds = xr.open_zarr(ds_path, consolidated=True)
        self.mean_std_dict = None
        self.patch_size = tuple(patch_size)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.stats_dir = stats_dir
        self.ds['sst'] = self.ds['sst'].where(self.ds['sst'] >= 0)
        if debug:
            self.num_timesteps = 5
        else:
            self.num_timesteps = -1

        self.num_timesteps = -1
    
    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            print(self.ds[self.input_vars])

            if not self.data_train and not self.data_val and not self.data_test:
                print(self.ds[self.input_vars])
            # IMPORTANT! Call sample_dataset with ds.copy(). xarray Datasets are mutable
            # train_patch_size = 160 if self.random_crop else 128
            # val_patch_size = 128
            if self.ds_path_global:
                self.global_ds = load_global_ds(self.ds_path_global, self.input_vars, self.log_transform_vars, self.target,
                                                self.target_shift)
            else:
                print('Warning: No global ds path provided. Using local input only...')
                self.global_ds = None


            # Create dataset objects
            train_batches = self.output_path +'/train/saved_batches_info.json'
            
            self.mean_std_dict = self.output_path +'/train/mean_std.json'

            val_batches = self.output_path +'/val/saved_batches_info.json'

            test_batches = self.output_path +'/test/saved_batches_info.json'


            self.data_train = BatcherDS(train_batches, input_vars=self.input_vars,
                                        positional_vars=self.positional_vars, target=self.target,
                                        mean_std_dict=self.mean_std_dict)
            self.data_val = BatcherDS(val_batches, input_vars=self.input_vars,
                                      positional_vars=self.positional_vars, target=self.target,
                                      mean_std_dict=self.mean_std_dict)
            self.data_test = BatcherDS(test_batches, input_vars=self.input_vars,
                                       positional_vars=self.positional_vars, target=self.target,
                                       mean_std_dict=self.mean_std_dict)


    def train_dataloader(self):
         return DataLoader(
             dataset=self.data_train,
             batch_size=self.hparams.batch_size,
             num_workers=self.hparams.num_workers,
             pin_memory=self.hparams.pin_memory,
             shuffle=True,
             persistent_workers=True
         )

    def val_dataloader(self):
         return DataLoader(
             dataset=self.data_val,
             batch_size=self.hparams.batch_size,
             num_workers=self.hparams.num_workers,
             pin_memory=self.hparams.pin_memory,
             shuffle=False,
             persistent_workers=True
         )

    def test_dataloader(self):
         return DataLoader(
             dataset=self.data_test,
             batch_size=self.hparams.batch_size,
             num_workers=self.hparams.num_workers,
             pin_memory=self.hparams.pin_memory,
             shuffle=False,
             persistent_workers=True
         )


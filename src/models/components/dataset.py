"""
Dataset for training prototype patch classification model on Cityscapes and SUN datasets
"""
import json
from typing import Any, List, Optional, Tuple

import cv2
from PIL import Image

import torch
from torchvision.datasets import VisionDataset
from torchvision import transforms
import os
import random

from .constant import SEASFIRE_ID_MAPPING, SEASFIRE_CATEGORIES
#from .settings import data_path, log

import numpy as np


def resize_label(label, size):
    """
    Downsample labels by nearest interpolation.
    Other nearest methods result in misaligned labels.
    -> F.interpolate(labels, shape, mode='nearest')
    -> cv2.resize(labels, shape, interpolation=cv2.INTER_NEAREST)
    """
    label = Image.fromarray(label.astype(float)).resize(size, resample=Image.NEAREST)
    return torch.LongTensor(np.asarray(label))


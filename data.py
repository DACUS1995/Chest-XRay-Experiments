import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader


class DataHandler:
	def __init__(self, run_config):
		raise Exception("Must implement!")
		self._training_dataset = None
		self._validation_dataset = None
		self._run_config = run_config

		self.load_datasets()
		
	def load_datasets(self):
		pass	

	def get_data_loaders(self) -> Tuple[DataLoader]:
		return (
			DataLoader(self._training_dataset, batch_size=self._run_config.batch_size, shuffle=True, num_workers=self._run_config.workers, pin_memory=True), 
			DataLoader(self._validation_dataset, batch_size=self._run_config.batch_size, shuffle=True, num_workers=self._run_config.workers, pin_memory=True)
		)

	def get_datasets(self) -> Tuple[Dataset]:
		return self._training_dataset, self._validation_dataset

	def get_datasets_sizes(self) -> Tuple[int]:
		return len(self._training_dataset), len(self._validation_dataset)


class CustomDataset(Dataset):
	def __init__(self):
		pass

	def __getitem__(self, idx):
		pass

	def __len__(self):
		pass
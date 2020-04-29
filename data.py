import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from PIL import Image

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
		self._training_dataset = CustomDataset("./dataset/CheXpert-v1.0-small", "train")
		self._validation_dataset = CustomDataset("./dataset/CheXpert-v1.0-small", "valid")

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
	def __init__(self, root_path, type):
		self.root_path = root_path
		self.type = type

		df = pd.read_csv(root_path + "/" + type + ".csv")
		df = df[df["Frontal/Lateral"] == "Frontal"]

		self.df = df

		self.transformers = {
			'train_transforms' : transforms.Compose([
				transforms.Resize((224,224)),
				#transforms.CenterCrop(224),
				transforms.RandomRotation(20),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			]),
			'test_transforms' : transforms.Compose([
				transforms.Resize((224,224)),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			]),
			'valid_transforms' : transforms.Compose([
				transforms.Resize((224,224)),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			])
		}

	def __getitem__(self, idx):
		record = self.df.iloc(idx)
		img_path = os.path.join(self.root, self.type, record["path"])
		image = Image.open(img_path)
		image = torch.from_numpy() 

		if self.transformers is not None:
			image = self.transformers[self.type + "_transforms"]

		return image

	def __len__(self):
		return len(self.df)
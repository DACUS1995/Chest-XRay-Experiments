import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from PIL import Image
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

criterion_weight = None

class DataHandler:
	def __init__(self, run_config):
		self._training_dataset = None
		self._validation_dataset = None
		self._run_config = run_config

		self.load_datasets()
		
	def load_datasets(self):
		self._training_dataset = CustomDataset("./dataset", "train")
		self._validation_dataset = CustomDataset("./dataset", "valid")

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
	def __init__(self, root_path, run_type):
		self.root_path = root_path
		self.run_type = run_type	

		
		df = pd.read_csv(root_path + "/CheXpert-v1.0-small/" + run_type + ".csv")
		self.label_col = df.columns[5:]
		df = df[df["Frontal/Lateral"] == "Frontal"]
		df[self.label_col] = df[self.label_col].fillna(0)
		# df[self.label_col] = df[self.label_col].dropna()
		df[self.label_col] = df[self.label_col].replace(-1,1) # U-Ones
		df = df.reset_index()

		criterion_weight = torch.tensor([df[col].sum()/df.shape[0] for col in self.label_col])

		self.df = df[:5000]

		self.transformers = {
			'train_transforms' : transforms.Compose([
				transforms.Resize((224,224)),
				#transforms.CenterCrop(224),
				# transforms.RandomRotation(20),
				# transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			]),
			'test_transforms' : transforms.Compose([
				transforms.Resize((224,224)),
				# transforms.CenterCrop(224),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.5330], std=[0.0349])
			]),
			'valid_transforms' : transforms.Compose([
				transforms.Resize((224,224)),
				# transforms.CenterCrop(224),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.5330], std=[0.0349])
			])
		}

	def __getitem__(self, idx):
		record = self.df.loc[idx]
		img_path = os.path.join(self.root_path, record["Path"])
		image = Image.open(img_path).convert('RGB')

		if self.transformers is not None:
			image = self.transformers[self.run_type + "_transforms"](image)

		labels = record[self.label_col].to_numpy().astype('float32') 

		return image, labels

	def __len__(self):
		return len(self.df)
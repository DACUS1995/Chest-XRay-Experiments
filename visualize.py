import numpy as np
import pandas as pd
import argparse
from argparse import Namespace
from collections import OrderedDict
import time
import os
import copy
import datetime
from tqdm import tqdm
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics

from run_builder import RunBuilder
from data import DataHandler
from data import criterion_weight
from config import Config
import utils
from models.test_model import TestModel
from cam import get_cam

torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cpu")

competition_columns = [2, 5, 6, 8, 10]
img_path = "sample.jpg"
classes = [
	'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
	'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
	'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
	'Fracture', 'Support Devices'
]


def main(args: Namespace) -> None:
	model_save_path = "./test_model.pt"
	model = TestModel()
	model.load_state_dict(torch.load(model_save_path))
	_ = model.eval()

	image = Image.open(img_path)
	image = image.resize((224,224)).convert('RGB')

	preprocess = transforms.Compose([
		transforms.Resize((224,224)),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	img_tensor = preprocess(image)

	model.to(device)
	img_tensor.to(device)

	features = []
	def hook_feature(module, input, output):
		features.append(output.data.cpu().numpy())

	# model.densenet121.features.denseblock4.denselayer16.conv2.register_forward_hook(hook_feature)
	model.densenet121.features.norm5.register_forward_hook(hook_feature)
	get_cam(model, features, img_tensor, classes, img_path)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	args = parser.parse_args()
	main(args)
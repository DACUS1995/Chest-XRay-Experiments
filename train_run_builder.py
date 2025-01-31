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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics

from run_builder import RunBuilder
from data import DataHandler
from data import criterion_weight
from config import Config
import utils
from models.test_model import TestModel

torch.manual_seed(0)
np.random.seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


competition_columns = [2, 5, 6, 8, 10]

def train(cfg) -> None:
	device = torch.device(cfg.device)

	runs = None
	if cfg.use_run_setup == True:
		runs = RunBuilder.get_runs(cfg.run_setup)
	else:
		runs = RunBuilder.get_runs(
			OrderedDict({
				"lr": [cfg.lr],
				"num_epochs": [cfg.num_epochs]
			})
		)
	assert runs != None


	data_handler = DataHandler(cfg)
	train_dataset, validation_dataset = data_handler.get_datasets()
	train_loader, validation_loader = data_handler.get_data_loaders()
	training_dataset_size, validation_dataset_size = data_handler.get_datasets_sizes()


	best_model_wts = None
	best_acc = 0.0
	best_config = None

	for run in runs:
		comment = f"Run setup -- {run}"
		print(comment)

		model = TestModel()
		model.to(device)
		optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
		
		# Check if resume
		if use_run_setup == False and resume = True:
			checkpoint = torch.load("./checkpoints/ckp.pt")
			model.load_state_dict(checkpoint["model_state_dict"])
			optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
			run.num_epochs -= checkpoint["epoch"]


		loss_criterion = torch.nn.BCELoss(size_average = True, weight=criterion_weight)

		log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		writer = SummaryWriter(log_dir)

		since = time.time()

		for epoch in range(run.num_epochs):
			print('Epoch {}/{}'.format(epoch, run.num_epochs))
			print('-' * 10)

			########### Training step ###########
			model.train()
			training_loss = []
			running_loss = 0.0
			auc_score = 0
			
			for i, data in enumerate(tqdm(train_loader, desc=f"Epoch [{epoch + 1}] progress")):

				x_batch, label_batch = data
				x_batch, label_batch = x_batch.to(device), label_batch.to(device)

				optimizer.zero_grad()
				outputs = model(x_batch)
				_, preds = torch.max(outputs, 1)

				loss = loss_criterion(outputs, label_batch)

				loss.backward()
				optimizer.step()
				
				# statistics
				running_loss += loss.item() * x_batch.size(0)
				auc_score = metrics.roc_auc_score(
					label_batch[:,competition_columns].detach().cpu().view((-1)).numpy(), 
					outputs[:,competition_columns].detach().cpu().view((-1)).numpy()
				)
				training_loss.append(loss.item())

				# # tensorboard logging
				# if i % 1000 == 0:
				# 	writer.add_scalar("Loss/train", running_loss / 1000, epoch * len(trainloader) + i)
				# 	writer.add_scalar("Accuracy/train", running_loss / 1000, epoch * len(trainloader) + i)


			epoch_loss = running_loss / training_dataset_size
			epoch_acc = auc_score / training_dataset_size

			# tensorboard logging
			writer.add_scalar("Loss/train", epoch_loss, epoch)
			writer.add_scalar("Accuracy/train", epoch_acc, epoch)

			print('Training step => Loss: {:.4f} AUC: {:.4f}'.format(
				epoch_loss, epoch_acc
			))


			########### Validation step ###########
			model.eval()
			validation_loss = []
			running_loss = 0.0

			for i, data in enumerate(validation_loader):
				with torch.no_grad():
					x_batch, label_batch = data
					x_batch, label_batch = x_batch.to(device), label_batch.to(device)

					outputs = model(x_batch)
					_, preds = torch.max(outputs, 1)
					loss = loss_criterion(outputs, label_batch)

					running_loss += loss.item() * x_batch.size(0)
					auc_score = metrics.roc_auc_score(
						label_batch[:, competition_columns].detach().cpu().view((-1)).numpy(), 
						outputs[:,competition_columns].detach().cpu().view((-1)).numpy()
					)
					validation_loss.append(loss.item())
			
			epoch_loss = running_loss / validation_dataset_size
			epoch_acc = auc_score / validation_dataset_size

			# tensorboard logging
			writer.add_scalar("Loss/validation", epoch_loss, epoch)
			writer.add_scalar("Accuracy/validation", epoch_acc, epoch)

			print('Evaluation step => Loss: {:.4f} AUC: {:.4f}'.format(
				epoch_loss, epoch_acc
			))

			#Save the best model based on accuracy
			if epoch_acc > best_acc:
				best_acc = epoch_acc
				best_config = f"{run}"
				best_model_wts = copy.deepcopy(model.state_dict())

			#Checkpoint
			torch.save({
				"epoch": epoch,
				"model_state_dict": model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict()
			}, "./checkpoints/ckp.pt")


		time_elapsed = time.time() - since
		print('Training complete in {:.0f}m {:.0f}s'.format(
			time_elapsed // 60, time_elapsed % 60
		))
		print('Best (so far) validation Acc: {:4f}'.format(best_acc))


	print('-' * 10)
	print('### Final results ###\n')
	print('Best validation Acc: {:4f}'.format(best_acc))
	print(f"Best configuration: {best_config}")

	model.load_state_dict(best_model_wts)
	return model


def main(args: Namespace) -> None:
	trained_model = train(Config)
	torch.save(trained_model.state_dict(), f"{trained_model.name}.pt")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	args = parser.parse_args()
	main(args)
import torch.nn as nn
import torchvision

num_classes = 14 

class TestModel(nn.Module):
	def __init__(self, out_size = num_classes):
		super().__init__()
		self.densenet121 = torchvision.models.densenet121(pretrained=True)
		num_ftrs = self.densenet121.classifier.in_features
		self.densenet121.classifier = nn.Sequential(
			nn.Linear(num_ftrs, out_size),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = self.densenet121(x)
		return x
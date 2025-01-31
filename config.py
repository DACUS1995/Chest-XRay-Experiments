from collections import OrderedDict

class Config:
	resume = False
	use_run_setup = False
	run_setup = OrderedDict({
		"lr": [0.01, 0.001],
		"num_epochs": [10]
	})

	device = "cuda"
	num_epochs = 5
	lr = 0.001
	batch_size = 24
	test_batch_size = 5
	workers = 2
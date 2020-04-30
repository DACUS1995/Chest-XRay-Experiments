from collections import OrderedDict

class Config:
	use_run_setup = True
	run_setup = OrderedDict({
		"lr": [0.01, 0.001],
		"num_epochs": [10]
	})

	device = "cuda"
	num_epochs = 50
	lr = 0.001
	batch_size = 6
	test_batch_size = 5
	workers = 2
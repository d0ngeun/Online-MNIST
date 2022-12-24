import torch

#meta data / parameters
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
batch_size = 100
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
device = ("cpu")
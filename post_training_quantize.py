import torch.distributed as dist
import torch.utils.data.distributed
from config import SearchConfig
import utils
import os

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'

# 0.config
config = SearchConfig()
device = torch.device("cuda:0")
model_load_path = 'searchs/cifar10/best.pth.tar'
model_log = 'searchs/cifar10/model.log'
weights_log = 'searchs/cifar10/weights.log'
ptq_log = 'searchs/cifar10/ptq.log'
dist.init_process_group("nccl", rank=0, world_size=1)

if os.path.exists(model_log):
    os.remove(model_log)
if os.path.exists(weights_log):
    os.remove(weights_log)

# 1.prepare the data
input_size, input_channels, n_classes, train_data = utils.get_data(
        config.dataset, config.data_path, cutout_length=0, validation=False)
n_train = len(train_data)
split = n_train // 2
train_dataset, val_dataset = torch.utils.data.random_split(train_data, [int(split), n_train-split])
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
valid_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.batch_size,
                                               sampler=train_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
valid_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=config.batch_size,
                                               sampler=valid_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
# 2.load the model
model_float_ddp = torch.load(model_load_path)
model_float = model_float_ddp.module
with open(model_log, 'a') as file:
    file.write(str(model_float))

with open(weights_log, 'a') as file:
    for n, p in model_float.named_parameters():
        file.write(str(n) + '\n')

# 3.ptq implementation

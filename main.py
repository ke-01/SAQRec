import argparse
from utils.Context import ContextManager, DatasetManager
from config import const
import models.Inputs as data_in

import torch
import numpy as np
import random 
import os
import subprocess
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
    
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
setup_seed(1)

parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str, help='experiment name', default='default')
parser.add_argument('--description', type=str, help='exp details, used for log name', default='default')
parser.add_argument('--workspace', type=str, default='./workspace')

parser.add_argument('--dataset_name', type=str, default='commercial')
parser.add_argument('--use_cpu', dest='use_gpu', action='store_false')

parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--model', type=str, help='which model to use', default='SAQRec')
parser.add_argument('--num_negs', type=int, help='# negtive samples for training', default=2)

parser.set_defaults(use_gpu=True)
parser.add_argument('--batch_size', type=int, help='training batch_size', default=512)

args = parser.parse_args()

if args.dataset_name == 'commercial':
    const.init_dataset_setting_commercial()
    data_in.init_data_attribute_ls('commercial')
elif args.dataset_name == 'kuairand':
    const.init_dataset_setting_kuairand()
    data_in.init_data_attribute_ls('kuairand')
else:
    raise ValueError(f'Not support dataset: {args.dataset_name}')

cm = ContextManager(args)
dm = DatasetManager(args)

trainer = cm.set_trainer(args, cm, dm)

trainer.train()
trainer.test()
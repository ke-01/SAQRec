# Implementation of SAQRec
This is the official implementation of the paper "SAQRec: Aligning Recommender Systems to User Satisfaction via Questionnaire Feedback" based on PyTorch.

## Overview

The main implementation of SAQRec can be found in the file `models/SAQRec.py`. 



## Reproduction
Check the following instructions for reproducing experiments.

### Experimental Setting
All the hyper-parameter settings of SAQRec on both datasets can be found in files `config/SAQRec_commercial.yaml` and `config/SAQRec_kuairand.yaml`.
The settings of two datasets can be found in file `config/const.py`.

### Dataset
Check folder `data` for details.

### Quick Start
#### 1. Download and process data
Place data files in the folder `data`.

#### 2. Satisfy the requirements
Our experiments were done with the following python packages:
```
python==3.8.18
torch==1.10.1
numpy==1.24.1
scikit-learn==1.3.1
tqdm==4.66.1
PyYAML==6.0.1
```

#### 3. Train and evaluate our model:
Run codes in command line:
```bash
# Base model
python main.py --name base --workspace ./workspace/base --gpu_id 0  --epochs 100 --model Base  --batch_size 512 --dataset_name kuairand

# propensity model 
python main.py --name pro --workspace ./workspace/pro --gpu_id 0  --epochs 100 --model Pro_Model  --batch_size 512 --dataset_name kuairand

# satisfaction model 
python main.py --name satis --workspace ./workspace/satis --gpu_id 0  --epochs 100 --model Satis_Model  --batch_size 512 --dataset_name kuairand

# SAQRec 
python main.py --name SAQRec --workspace ./workspace/SAQRec --gpu_id 0  --epochs 100 --model SAQRec  --batch_size 512 --dataset_name kuairand
```

#### 4. Check training and evaluation process:
After training, check log files, for example, `workspace/SAQRec/log/default.log`.


### Environments
We conducted the experiments based on the following environments:
* CUDA Version: 11.1
* OS: CentOS Linux release 7.4.1708 (Core)
* GPU: The NVIDIA® T4 GPU
* CPU: Intel(R) Xeon(R) Gold 6230R CPU @ 2.10GHz
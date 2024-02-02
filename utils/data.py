import torch
import random
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from .data_sampler import *
from .data_utils import *

class BaseDataset(Dataset):
    
    def __init__(self):
        super(Dataset, self).__init__()


    def __len__(self):

        return self.sampler.record.shape[0]

    def __getitem__(self, index):

        raise NotImplementedError


class Pointwise_Dataset(BaseDataset):
    def __init__(self, dataset_file, flags_obj, is_train):
        super().__init__()
        self.sampler = Point_Sampler(dataset_file, flags_obj, is_train=is_train)  

    def __getitem__(self, index):
        '''
        Args: 
            index: index of interaction file
        Returns:
            users: list of user attributes
            items: list of item attributes
            rec_hiss: list of attributes of items in recommendation history,
            labels: list of labels, shape (1+num_neg)
        '''
        users, rec_hiss, satis_hiss,dissatis_his, pos_item, neg_items,satis = self.sampler.sample(index)
        return users, rec_hiss, satis_hiss,dissatis_his, pos_item, neg_items,satis

class SAQRec_Dataset(Pointwise_Dataset):
    def __init__(self, *args):
        super().__init__(*args)
class Base_Dataset(Pointwise_Dataset):
    def __init__(self, *args):
        super().__init__(*args)
class Pro_Model_Dataset(Pointwise_Dataset):
    def __init__(self, *args):
        super().__init__(*args)
class Satis_Model_Dataset(Pointwise_Dataset):
    def __init__(self, *args):
        super().__init__(*args)

GLOBAL_SEED = 1
 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
 
GLOBAL_WORKER_ID = None
def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)

def get_dataloader(data_set, bs, **kwargs):
    return DataLoader(  data_set, batch_size = bs,
                        shuffle=False, pin_memory = True, 
                        worker_init_fn=worker_init_fn, **kwargs
                    )


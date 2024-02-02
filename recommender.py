

import torch
import torch.nn as nn
import torch.optim as optim

from models import *

import utils.data as data
from utils import Context as ctxt
import config.const as const_util

import os
import yaml

class Recommender(object):

    def __init__(self, flags_obj, workspace, dm, nc=None):

        self.dm = dm # dataset manager
        self.model_name = flags_obj.model
        self.flags_obj = flags_obj
        self.set_device()
        self.load_model_config()
        self.set_model()
        self.workspace = workspace

    def set_device(self):

        self.device  = ctxt.ContextManager.set_device(self.flags_obj)

    def load_model_config(self):
        path = 'config/{}_{}.yaml'.format(self.model_name, self.dm.dataset_name)
        f = open(path)
        self.model_config = yaml.load(f, Loader=yaml.FullLoader)

    def set_model(self):

        raise NotImplementedError

    def transfer_model(self):

        self.model = self.model.to(self.device)

    def save_ckpt(self):

        ckpt_path = os.path.join(self.workspace, const_util.ckpt)
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)

        model_path = os.path.join(ckpt_path, 'best.pth')
        torch.save(self.model.state_dict(), model_path)

    def load_ckpt(self, assigned_path=None):

        ckpt_path = os.path.join(self.workspace, const_util.ckpt)
        model_path = None
        if assigned_path is not None:
            '''specific assigned path'''
            model_path = assigned_path
        else:
            '''default path'''   
            model_path = os.path.join(ckpt_path, 'best.pth')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def get_dataset(self, *args):

        return getattr(data, f'{self.model_name.upper()}_Dataset')(*args)

    def get_optimizer(self):

        return optim.Adam(self.model.parameters(), lr=self.model_config['lr'], weight_decay=self.model_config['weight_decay'])

    def predict(self, sample):
        '''sample: input data and labels'''
        sample = [[k.to(self.device) for k in i] if type(i) == list else i.to(self.device) for i in sample]
        input_data = sample[:-2] 
        return self.model.predict(input_data) 

    def get_loss(self, sample,epoch):
        '''sample: input data and labels'''
        sample = [[k.to(self.device) for k in i] if type(i) == list else i.to(self.device) for i in sample]
        return self.model.forward(sample,epoch)


class SAQRec_Recommender(Recommender):
    def __init__(self, flags_obj, workspace, dm, nc):
        super().__init__(flags_obj, workspace, dm, nc)
    def set_model(self):
        self.model = SAQRec(config=self.model_config)
        
class Base_Recommender(Recommender):
    def __init__(self, flags_obj, workspace, dm, nc):
        super().__init__(flags_obj, workspace, dm, nc)
    def set_model(self):
        self.model = Base(config=self.model_config)
        
class Pro_Model_Recommender(Recommender):
    def __init__(self, flags_obj, workspace, dm, nc):
        super().__init__(flags_obj, workspace, dm, nc)
    def set_model(self):
        self.model = Pro_Model(config=self.model_config)
        
class Satis_Model_Recommender(Recommender):
    def __init__(self, flags_obj, workspace, dm, nc):
        super().__init__(flags_obj, workspace, dm, nc)
    def set_model(self):
        self.model = Satis_Model(config=self.model_config)
        
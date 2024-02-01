#coding=utf-8

import os
import datetime

import logging
import torch

import config.const as const_util
import trainer
import recommender

import datetime

class ContextManager(object):

    def __init__(self, flags_obj):

        self.exp_name = flags_obj.name
        self.description = flags_obj.description
        self.workspace_root = flags_obj.workspace
        self.set_workspace(flags_obj)
        self.set_logging(flags_obj)


    def set_workspace(self, flags_obj):

        if not os.path.exists(self.workspace_root):
            os.mkdir(self.workspace_root)

        self.workspace = self.workspace_root
        if not os.path.exists(self.workspace):
            os.mkdir(self.workspace)
    
    def set_logging(self, flags_obj):
        # set log file path
        if not os.path.exists(os.path.join(self.workspace, 'log')):
            os.mkdir(os.path.join(self.workspace, 'log'))
        log_file_name = os.path.join(self.workspace, 'log', self.description+'.log')
        logging.basicConfig(format='%(asctime)s - %(message)s',
                    level=logging.INFO, filename=log_file_name, filemode='w')
        
    @staticmethod
    def set_trainer(flags_obj, cm,  dm, nc=None):

        Sequence_Trainer_model = ['Base','SAQRec','Pro_Model','Satis_Model']
        if flags_obj.model.upper() in Sequence_Trainer_model:
            return getattr(trainer, f'{flags_obj.model.upper()}_Trainer')(flags_obj, cm, dm, nc)
        else:
            print()
            raise NameError('trainer model name error!')

    @staticmethod
    def set_recommender(flags_obj, workspace, dm, new_config):

        rec = getattr(
            recommender, '_'.join(
                [flags_obj.model.upper(), 'Recommender']
                )
            ) (flags_obj, workspace, dm, new_config)
          
        return rec

    @staticmethod
    def set_device(flags_obj):

        if not flags_obj.use_gpu:
            return torch.device('cpu')
        else:
            return torch.device('cuda')



class DatasetManager(object):

    def __init__(self, flags_obj):

        self.dataset_name = flags_obj.dataset_name
        self.batch_size = flags_obj.batch_size
        self.num_negs = flags_obj.num_negs
        self.set_dataset()


    def set_dataset(self):
        
        data_manager_attr = ['load_path', 'user_vocab', 'item_vocab']
        for attr in data_manager_attr:
            setattr(self, attr, getattr(const_util, attr))

        logging.info('dataset: {}'.format(self.dataset_name))

    def show(self):
        print(self.__dict__)




class EarlyStopManager(object):

    def __init__(self, config):

        self.min_lr = config['min_lr']
        self.es_patience = config['es_patience']
        self.count = 0
        self.max_metric = 0

    def step(self, lr, metric):

        if lr > self.min_lr:
            if metric > self.max_metric:
                self.max_metric = metric
            return False
        else:
            if metric > self.max_metric:
                self.max_metric = metric
                self.count = 0
                return False
            else:
                self.count = self.count + 1
                if self.count > self.es_patience:
                    return True
                return False



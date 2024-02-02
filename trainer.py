import os
from tqdm import tqdm
from utils.data import get_dataloader
import config.const as const_util

from utils import Context as ctxt
from tester import Tester
# from tester_ovo import Tester

import torch
import torch.optim as optim

import logging


class Trainer(object):

    def __init__(self, flags_obj, cm,  dm, new_config=None):
        """
        Args:
            flags_obj: arguments in main.py
            cm : context manager
            dm : dataset manager
            new config : update default model config(`./config/model_kuaishou.yaml`) to tune hyper-parameters
        """

        self.name = flags_obj.name + '_trainer'
        self.cm = cm #context manager
        self.dm = dm #dataset manager
        self.flags_obj = flags_obj
        self.set_recommender(flags_obj, cm.workspace, dm, new_config)
        self.recommender.transfer_model()
        self.lr = self.recommender.model_config['lr']
        self.tester = Tester(flags_obj, self.recommender)
        

    def set_recommender(self, flags_obj, workspace, dm, new_config):

        self.recommender = ctxt.ContextManager.set_recommender(flags_obj, workspace, dm, new_config)

    def train(self):

        self.set_dataloader()
        self.set_optimizer()
        self.set_scheduler()
        self.set_esm() #early stop manager

        best_metric = 0
        train_loss = [0.0, 0.0, 0.0, 0.0] #store every training loss
        val_loss = [0.0]
        
        for epoch in range(self.flags_obj.epochs):
            print('epoch:{}'.format(epoch))
            self.train_one_epoch(epoch, train_loss)
            watch_metric_value = self.validate(epoch, val_loss)
            if watch_metric_value > best_metric:
                self.recommender.save_ckpt()
                logging.info('save ckpt at epoch {}'.format(epoch))
                best_metric = watch_metric_value
            self.scheduler.step(watch_metric_value)

            stop = self.esm.step(self.lr, watch_metric_value)
            if stop:
                break

    def set_test_dataloader(self):
        raise NotImplementedError

    def test(self, assigned_model_path = None, load_config=True):
        '''
            test model on test dataset
        Args:
            tune_para
        '''

        self.set_test_dataloader()
        
        if load_config:
            self.recommender.load_ckpt(assigned_path = assigned_model_path)

        results = self.tester.test()

        logging.info('TEST results :')
        self.record_metrics('test', results)
        print('test: ', results)    

    def set_dataloader(self):

        raise NotImplementedError

    def set_optimizer(self):

        self.optimizer = self.recommender.get_optimizer()

    def set_scheduler(self):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
         mode='max', patience=self.recommender.model_config['patience'], 
         min_lr=self.recommender.model_config['min_lr'])

    def set_esm(self):

        self.esm = ctxt.EarlyStopManager(self.recommender.model_config)



    def record_metrics(self, epoch, metric):
        """
        record metrics after each epoch
        """    

        logging.info('VALIDATION epoch: {}, results: {}'.format(epoch, metric))

    def train_one_epoch(self, epoch, train_loss):

        self.lr = self.train_one_epoch_core(epoch, self.dataloader, self.optimizer, self.lr, train_loss)

    def train_one_epoch_core(self, epoch, dataloader, optimizer, lr, train_loss):

        epoch_loss = train_loss[0]

        self.recommender.model.train()
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < lr:

            lr = current_lr
            logging.info('reducing learning rate!')

        logging.info('learning rate : {}'.format(lr))

        tqdm_ = tqdm(iterable=dataloader, mininterval=1, ncols=100)
        for step, sample in enumerate(tqdm_):

            optimizer.zero_grad()
            loss = self.recommender.get_loss(sample,epoch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            print("epoch {:d} , step {:d} , loss: {:.4f}".format(epoch+1, step+1, epoch_loss / (step+1+epoch*dataloader.__len__())))
        logging.info('epoch {}:  loss = {}'.format(epoch, epoch_loss/(step+1+epoch*dataloader.__len__())))

        train_loss[0] = epoch_loss

        return lr

    def validate(self, epoch, total_loss):

        results = self.tester.test()
        self.record_metrics(epoch, results)
        print(results)
       
        return results['mrr']


class SAQRec_Trainer(Trainer):

    def __init__(self, flags_obj, cm, dm, nc):
        
        super().__init__(flags_obj, cm, dm, nc)

    def set_dataloader(self):
    
        # training dataloader
        dst = self.recommender.get_dataset(const_util.train_file, self.dm, True)
        self.dataloader = get_dataloader(
            data_set = dst,
            bs = self.dm.batch_size,
            prefetch_factor = self.dm.batch_size // 32 + 1, num_workers = 32,
        )
        # validation dataloader
        self.tester.set_dataloader(
            dst = self.recommender.get_dataset(const_util.valid_file, self.dm, False), #这里和train的区别是第三项是false,is_train区别
        )

    def set_test_dataloader(self):

        dst = self.recommender.get_dataset(const_util.test_file, self.dm, False)
        self.tester.set_dataloader(
            dst = dst,
        )



from utils.metrics import judger as judge
from utils.data import get_dataloader

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np


class Tester(object):

    def __init__(self, flags_obj, recommender):

        self.recommender = recommender
        self.flags_obj = flags_obj
        self.judger = judge()
        self.results = {}


    def set_dataloader(self, dst, **kwargs):

        self.num_neg = 99
        self.batch_size =  (self.num_neg+1) * 50 # multiple of group_size. for kuaishou dataset
        self.dataloader =   get_dataloader(
                                dst, self.batch_size,
                                prefetch_factor = self.batch_size // 16 * 2, num_workers = 16,
                                **kwargs
                            )


    @torch.no_grad()
    def test(self):

        self.recommender.model.eval()

        group_preds, group_labels = self._run_eval(
            group_size = (1 + self.num_neg)
        )
        
        res_pairwise = self.judger.cal_metric(
            group_preds, group_labels, ['ndcg@5;10', 'hit@1;5;10','mrr']
        )

        self.results.update(res_pairwise)

        return self.results

    def _run_eval(self, group_size):
        """ making prediction and group results

        Args:
            group_size: 100 for validation/test set(# negative samples is 99); 
        """
        group_preds, group_labels = [],[]

        for batch_data in tqdm(iterable=self.dataloader, mininterval=1, ncols=100):
            step_pred = self.recommender.predict(batch_data).squeeze(-1).cpu().numpy()
            step_label = batch_data[-2].cpu().numpy()
            group_preds.extend(np.reshape(step_pred, (-1, group_size)))
            group_labels.extend(np.reshape(step_label, (-1, group_size)))

        # cpu() results in gpu memory not auto-collected
        # this command frees memory in Nvidia-smi
        if self.recommender.device != torch.device('cpu'):
            torch.cuda.empty_cache() 
        
        return group_preds, group_labels

   
      

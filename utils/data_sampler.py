from .data_utils import NpyLoader, TsvLoader, JsonLoader, PickleLoader
import random
from config import const
import torch
import numpy as np


class Sampler(object):
    
    def __init__(self, dataset_file, load_path):

        self.load_path = load_path
        self.tsv_loader = TsvLoader(load_path)
        self.json_loader = JsonLoader(load_path)
        self.pickle_loader = PickleLoader(load_path)

        # user item interactions
        self.record = self.tsv_loader.load(filename=dataset_file, sep='\t')
    
    def sample(self, index, **kwargs):

        raise NotImplementedError


class Point_Sampler(Sampler):
    def __init__(self, dataset_file, args, is_train):
        super().__init__(dataset_file, args.load_path)
        self.is_train = is_train
        self.dataset_name = args.dataset_name
        self.build(args)

    def build(self, args):
        self.model=args.model
        self.num_negs = 0 if not self.is_train else args.num_negs
        self.sample = self.train_sample if self.is_train else self.test_sample

        self.user_vocab = self.pickle_loader.load(args.user_vocab)
        self.item_vocab = self.pickle_loader.load(args.item_vocab)

        self.items_with_popular = self.record['i_id'].tolist() 
        self.max_rec_his = const.max_rec_his_len
        self.max_satis_his = const.max_satis_his_len
        self.max_dissatis_his = const.max_dissatis_his_len

        if self.dataset_name=='commercial':
            self._get_user_profile = self.commercial_get_user_profile
            self._get_item_info = self.commercial_get_item_info 
            self.parse_line = self.commercial_parse_line 
        elif self.dataset_name=='kuairand':
            self._get_user_profile = self._kuairand_get_user_profile
            self._get_item_info = self._kuairand_get_item_info 
            self.parse_line = self._kuairand_parse_line 


    def commercial_get_user_profile(self, user):
        
        return (self.user_vocab[user]['id'], self.user_vocab[user]['gender'], self.user_vocab[user]['age'], \
            self.user_vocab[user]['src_level'])
    def _kuairand_get_user_profile(self, user):
        
        return (self.user_vocab[user]['id'], self.user_vocab[user]['gender'], self.user_vocab[user]['age_range'],
        self.user_vocab[user]['fre_city_level'], self.user_vocab[user]['fre_country_region'], self.user_vocab[user]['user_active_degree'])

    def commercial_get_item_info(self, item):
        if item == 0:
            # skip padding
            return (0, 0, 0)
        
        return (
            self.item_vocab[item]['id'],  self.item_vocab[item]['type1'], \
            self.item_vocab[item]['cate']
        )
    def _kuairand_get_item_info(self, item):
        if item == 0:
            # skip padding
            return (0, 0, 0,0)

        return (
            self.item_vocab[item]['id'],  self.item_vocab[item]['first_level_category_id'],self.item_vocab[item]['second_level_category_id'], \
            self.item_vocab[item]['province_name']
        )

    def _gen_neg_samples(self, postive_item, user_rec_his):
        count = 0
        neg_items = []
        while count < self.num_negs:
            neg_item = random.choice(self.items_with_popular) 
            if  neg_item == postive_item or\
                neg_item in neg_items or\
                neg_item in user_rec_his:
                continue
            count += 1
            neg_items.append(neg_item)
            
        return neg_items

    def commercial_parse_line(self, index):

        line = self.record.iloc[index] 

        user = int(line['u_id'])
        pos_item = int(line['i_id'])
        rec_his_num = int(line['rec_his'])
        src_his_num = int(line['src_his'])
        label = float(line['label'])

        return user, pos_item, rec_his_num, src_his_num, label
    def _kuairand_parse_line(self, index):

        line = self.record.iloc[index] 

        user = int(line['u_id'])
        pos_item = int(line['i_id'])
        time_ms = int(line['time_ms'])
        click_cnt=int(line['count_click'])
        satis_cnt=int(line['cnt_satis'])
        dissatis_cnt=int(line['cnt_dissatis'])
        if self.model == 'Pro_Model' :
            labels =  0 if float(line['satis']) == 0.5 else 1
        elif self.model == 'Satis_Model':
            labels =  float(line['satis']) 
        else : #base and SAQRec
            labels = [float(line['click'])]
        if 'satis' in line.keys():
            satis = [float(line['satis'])]
        else: satis=0.5

        return user, pos_item, time_ms, click_cnt, satis_cnt,dissatis_cnt,labels,satis
    def wrap_item_his_ls(self, item_ls):
        '''
        Args:
            item_ls: list of item original IDs
        '''
        item_ls = list(
           torch.tensor(list(elem), dtype=torch.int64) for elem in zip(*[self._get_item_info(it)  for it in item_ls])
        )

        return item_ls

    def train_sample(self, index):
        
        if self.dataset_name=='commercial':
            user, pos_item, time_ms, click_cnt, satis_cnt,dissatis_cnt,labels,satis = self.parse_line(index)

            rec_his = self.user_vocab[user]['click_his'][:click_cnt] 
            satis_his = self.user_vocab[user]['satis_his'][:satis_cnt] 
            dissatis_his = self.user_vocab[user]['dissatis_his'][:dissatis_cnt] 

            neg_items = []
            if self.num_negs > 0:
                neg_items = self._gen_neg_samples(pos_item, rec_his)
            neg_items = self.wrap_item_his_ls(neg_items) 
            pos_item = list(self._get_item_info(pos_item))
            users = [ attr for attr in self._get_user_profile(user) ]

            rec_his = rec_his[-self.max_rec_his:]
            if len(rec_his) < self.max_rec_his:
                rec_his += [0]*(self.max_rec_his - len(rec_his))
            rec_hiss = self.wrap_item_his_ls(rec_his) 
            satis_his = satis_his[-self.max_satis_his:]
            if len(satis_his) < self.max_satis_his:
                satis_his += [0]*(self.max_satis_his - len(satis_his))
            satis_hiss = self.wrap_item_his_ls(satis_his) 
            dissatis_his = dissatis_his[-self.max_dissatis_his:]
            if len(dissatis_his) < self.max_dissatis_his:
                dissatis_his += [0]*(self.max_dissatis_his - len(dissatis_his))
            dissatis_hiss = self.wrap_item_his_ls(dissatis_his) 

            return users, rec_hiss, satis_hiss,dissatis_hiss,pos_item, neg_items,satis
        elif self.dataset_name=='kuairand':
            user, pos_item, play_time_ms, click_cnt, satis_cnt,dissatis_cnt,labels,satis = self.parse_line(index)
            rec_his = self.user_vocab[user]['click_his'][:click_cnt] 
            plays= self.user_vocab[user]['play_time_ms'][:click_cnt]
            satis_his = self.user_vocab[user]['satis_his'][:satis_cnt] 
            dissatis_his = self.user_vocab[user]['dissatis_his'][:dissatis_cnt] 

            neg_items = []
            if self.num_negs > 0:
                neg_items = self._gen_neg_samples(pos_item, rec_his)  
            neg_items = self.wrap_item_his_ls(neg_items) 
            pos_item = list(self._get_item_info(pos_item))
            users = [ attr for attr in self._get_user_profile(user) ]

            rec_his = rec_his[-self.max_rec_his:]
            plays=plays[-self.max_rec_his:]
            if len(dissatis_his)==0:
                min_index = plays.index(min(plays))
                dissatis_his.append(rec_his[min_index])  
            if len(satis_his)==0:
                max_index = plays.index(max(plays))
                satis_his.append(rec_his[max_index])  
            if len(rec_his) < self.max_rec_his:
                rec_his += [0]*(self.max_rec_his - len(rec_his))
            rec_hiss = self.wrap_item_his_ls(rec_his) 
            satis_his = satis_his[-self.max_satis_his:]
            if len(satis_his) < self.max_satis_his:
                satis_his += [0]*(self.max_satis_his - len(satis_his))
            satis_hiss = self.wrap_item_his_ls(satis_his) 
            dissatis_his = dissatis_his[-self.max_dissatis_his:]
            if len(dissatis_his) < self.max_dissatis_his:
                dissatis_his += [0]*(self.max_dissatis_his - len(dissatis_his))
            dissatis_hiss = self.wrap_item_his_ls(dissatis_his)


            return users, rec_hiss, satis_hiss,dissatis_hiss,pos_item, neg_items,satis 
    def test_sample(self, index):
        if self.dataset_name=='commercial':
            user, pos_item, time_ms, click_cnt, satis_cnt,dissatis_cnt,labels,satis  = self.parse_line(index)

            rec_his = self.user_vocab[user]['click_his'][:click_cnt] 
            satis_his = self.user_vocab[user]['satis_his'][:satis_cnt] 
            dissatis_his = self.user_vocab[user]['dissatis_his'][:dissatis_cnt] 
            item = list(self._get_item_info(pos_item))
            users = [ attr for attr in self._get_user_profile(user) ]

            rec_his = rec_his[-self.max_rec_his:]
            if len(rec_his) < self.max_rec_his:
                rec_his += [0]*(self.max_rec_his - len(rec_his))
            rec_hiss = self.wrap_item_his_ls(rec_his)
            satis_his = satis_his[-self.max_satis_his:]
            if len(satis_his) < self.max_satis_his:
                satis_his += [0]*(self.max_satis_his - len(satis_his))
            satis_hiss = self.wrap_item_his_ls(satis_his) 
            dissatis_his = dissatis_his[-self.max_dissatis_his:]
            if len(dissatis_his) < self.max_dissatis_his:
                dissatis_his += [0]*(self.max_dissatis_his - len(dissatis_his))
            dissatis_hiss = self.wrap_item_his_ls(dissatis_his) 
            labels = torch.tensor(labels, dtype=torch.float32)

            return users, rec_hiss, satis_hiss,dissatis_hiss, item, labels,satis
        elif self.dataset_name=='kuairand':
            user, pos_item, play_time_ms, click_cnt, satis_cnt,dissatis_cnt,labels,satis  = self.parse_line(index)

            rec_his = self.user_vocab[user]['click_his'][:click_cnt] 
            plays= self.user_vocab[user]['play_time_ms'][:click_cnt]
            satis_his = self.user_vocab[user]['satis_his'][:satis_cnt] 
            dissatis_his = self.user_vocab[user]['dissatis_his'][:dissatis_cnt] 
            item = list(self._get_item_info(pos_item))
            users = [ attr for attr in self._get_user_profile(user) ]

            rec_his = rec_his[-self.max_rec_his:]
            plays=plays[-self.max_rec_his:]
            if len(dissatis_his)==0:
                min_index = plays.index(min(plays))
                dissatis_his.append(rec_his[min_index]) 
            if len(satis_his)==0:
                max_index = plays.index(max(plays))
                satis_his.append(rec_his[max_index])  
            if len(rec_his) < self.max_rec_his:
                rec_his += [0]*(self.max_rec_his - len(rec_his))
            rec_hiss = self.wrap_item_his_ls(rec_his)
            satis_his = satis_his[-self.max_satis_his:]
            if len(satis_his) < self.max_satis_his:
                satis_his += [0]*(self.max_satis_his - len(satis_his))
            satis_hiss = self.wrap_item_his_ls(satis_his) 
            dissatis_his = dissatis_his[-self.max_dissatis_his:]
            if len(dissatis_his) < self.max_dissatis_his:
                dissatis_his += [0]*(self.max_dissatis_his - len(dissatis_his))
            dissatis_hiss = self.wrap_item_his_ls(dissatis_his) 
            labels = torch.tensor(labels, dtype=torch.float32)

            return users, rec_hiss, satis_hiss,dissatis_hiss, item, labels,satis

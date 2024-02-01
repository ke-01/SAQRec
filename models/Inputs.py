import torch.nn as nn
import torch

from config import const

def init_data_attribute_ls(dataset_name):
    global user_attr_ls, item_attr_ls
    if dataset_name == 'kuaishou':
        user_attr_ls = ['id', 'gender', 'age', 'src_level']
        item_attr_ls = ['id', 'type1', 'cate']
    elif dataset_name == 'amazon':
        user_attr_ls = ['id']
        item_attr_ls = ['id']
    elif dataset_name=='commercial':
        user_attr_ls = ['id', 'gender', 'age_range','fre_city_level','fre_country_region','user_active_degree']
        item_attr_ls = ['id', 'first_level_category_id', 'second_level_category_id','province_name']


class user_feat(nn.Module):
    def __init__(self):
        super().__init__()

        global user_attr_ls
        self.attr_ls = user_attr_ls

        self.size = 0
        for attr in self.attr_ls: 
            setattr(
                self, f'user_{attr}_emb', 
                nn.Embedding(
                    num_embeddings = getattr(const, f'user_{attr}_num'),
                    embedding_dim = getattr(const, f'user_{attr}_dim')
                )
            )
            self.size += getattr(const, f'user_{attr}_dim')

    def get_emb(self, sample):
        feats_ls = []
        for i, attr in enumerate(sample):
            feats_ls.append(
                getattr(self, f'user_{self.attr_ls[i]}_emb')(attr)
            ) 
        return torch.cat(feats_ls, dim=-1)



class item_feat(nn.Module):
    def __init__(self):
        super().__init__()

        global item_attr_ls
        self.attr_ls = item_attr_ls

        self.size = 0
        for attr in self.attr_ls:
            setattr(
                self, f'item_{attr}_emb', 
                nn.Embedding(
                    num_embeddings = getattr(const, f'item_{attr}_num'),
                    embedding_dim = getattr(const, f'item_{attr}_dim'),
                    padding_idx = 0 if attr in ['id'] else None
                )
            )
            self.size += getattr(const, f'item_{attr}_dim')
        
    def get_emb(self, sample):
        feats_ls = []
        for i, attr in enumerate(sample):
            feats_ls.append(
                getattr(self, f'item_{self.attr_ls[i]}_emb')(attr)
            ) 
        return torch.cat(feats_ls, dim=-1)
 
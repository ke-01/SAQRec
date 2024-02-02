import torch
import torch.nn as nn
from config import const
import models.Inputs as data_in
from .module import FullyConnectedLayer,SharedBottomModel2
from .Inputs import user_feat, item_feat
from .Pro_Model import Pro_Model
import yaml
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
device = torch.device("cuda")

pre_trained_model='../workspace/base/ckpt/best.pth'
pro_model_path='../workspace/pro/ckpt/best.pth'
pro_config = yaml.load(open('config/Pro_Model_kuairand.yaml'), Loader=yaml.FullLoader)
pro_model=Pro_Model(pro_config)
pro_model.load_state_dict(torch.load(pro_model_path, map_location=device))
pro_model.to(device)
pro_model.eval()
for param in pro_model.parameters():
    param.requires_grad = False

class Satis_Model(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.user_feat = user_feat()
        self.item_feat = item_feat()

        self.drop_out=config['drop_out']
        self.clamp=config['clamp']
        pred_hid_units=config['hid_units']

        self.fc_layer = FullyConnectedLayer(input_size = self.item_feat.size+self.user_feat.size,
                    hidden_unit=pred_hid_units,
                    batch_norm=False,
                    sigmoid = True,
                    activation='relu',
                    dropout=self.drop_out,
                    )
        
        self.loss_func = nn.BCELoss(reduction='none')

        self._init_weights()
        
        all_params = torch.load(pre_trained_model,map_location=device)

        desired_params = {}
        for name, param in all_params.items():
            if 'emb' in name :  
                desired_params[name] = param
        result=self.load_state_dict(desired_params, strict=False)

        all_params = dict(self.named_parameters())
        for name, param in all_params.items():
            if 'emb' in name :
                param.requires_grad = False

        
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m)


    def input_from_feature_tables(self, input_data):

        user, rec_his,satis_his,dissatis_his, pos_item, neg_items  = input_data


        user_emb = self.user_feat.get_emb(user)
        pos_item_emb = self.item_feat.get_emb(pos_item)
        neg_item_emb = self.item_feat.get_emb(neg_items)

        rec_his_emb = self.item_feat.get_emb(rec_his)  
        rec_his_mask = torch.where(
                            rec_his[0]==0,
                            1, 0).bool()
        satis_his_emb = self.item_feat.get_emb(satis_his)
        satis_his_mask = torch.where(
                            satis_his[0]==0,
                            1, 0).bool()
        dissatis_his_emb = self.item_feat.get_emb(dissatis_his)
        dissatis_his_mask = torch.where(
                            dissatis_his[0]==0,
                            1, 0).bool()

        return user_emb, pos_item_emb,neg_item_emb, rec_his_emb, rec_his_mask,satis_his_emb,satis_his_mask,dissatis_his_emb,dissatis_his_mask
    def input_from_feature_tables_test(self, input_data):
        user, rec_his,satis_his,dissatis_his, pos_item  = input_data

        user_emb = self.user_feat.get_emb(user)
        pos_item_emb = self.item_feat.get_emb(pos_item)
 
        rec_his_emb = self.item_feat.get_emb(rec_his)  
        rec_his_mask = torch.where(
                            rec_his[0]==0,
                            1, 0).bool()
        satis_his_emb = self.item_feat.get_emb(satis_his)
        satis_his_mask = torch.where(
                            satis_his[0]==0,
                            1, 0).bool()
        dissatis_his_emb = self.item_feat.get_emb(dissatis_his)
        dissatis_his_mask = torch.where(
                            dissatis_his[0]==0,
                            1, 0).bool()

        return user_emb, pos_item_emb, rec_his_emb, rec_his_mask,satis_his_emb,satis_his_mask,dissatis_his_emb,dissatis_his_mask


    def forward(self, input_data,labels=None):
        user_emb, item_emb,neg_item_emb, rec_his_emb, rec_his_mask,satis_his_emb,satis_his_mask,dissatis_his_emb,dissatis_his_mask = self.input_from_feature_tables(input_data=input_data)
        
        emb = torch.cat([item_emb,  user_emb], dim=1)

        logits=(self.fc_layer(emb).squeeze(1))
        if labels is not None:            
            with torch.no_grad():
                pro=pro_model(input_data)
                pro = torch.clamp(pro, min=self.clamp)
            loss=self.loss_func(logits, labels)
            loss/=pro

            loss=loss.mean() 

            return loss
        else:
            return logits

import torch
import torch.nn as nn
from .module import *
from .Satis_Model import Satis_Model
from .Inputs import user_feat, item_feat
from config import const
import torch.nn.functional as F
import copy
import os
import yaml

os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
device = torch.device("cuda")
pre_trained_model='../workspace/base/ckpt/best.pth'
satis_model_path='../workspace/satis/ckpt/best.pth'
satis_config = yaml.load(open('config/Satis_Model_kuairand.yaml'), Loader=yaml.FullLoader)
satis_model=Satis_Model(satis_config)
satis_model.load_state_dict(torch.load(satis_model_path, map_location=device))
satis_model.to(device)
satis_model.eval()

class SAQRec(torch.nn.Module):
    """
    A pytorch implementation of SAQRec.
    """

    def __init__(self, config):
        super().__init__()        
        self.user_feat = user_feat()
        self.item_feat = item_feat()
        self.trans_dropout=config['trans_drop']
        self.mlp_dropout=config['mlp_drop']
        self.epoch_correct=config['epoch_correct']
        self.weight_2=config['weight_2']
        self.num_interest=config['num_interest']
        
        self.transformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.item_feat.size,
            nhead=2,
            dim_feedforward=self.item_feat.size,
            dropout=self.trans_dropout,
            batch_first=True)
        self.rec_his_transformer_layer = nn.TransformerEncoder(
            self.transformerEncoderLayer, num_layers=1)
        
        self.transformerEncoderLayer_satis = nn.TransformerEncoderLayer(
            d_model=self.item_feat.size,
            nhead=2,
            dim_feedforward=self.item_feat.size,
            dropout=self.trans_dropout,
            batch_first=True)
        self.satis_his_transformer_layer = nn.TransformerEncoder(
            self.transformerEncoderLayer_satis, num_layers=1)
        
        self.transformerEncoderLayer_dissatis = nn.TransformerEncoderLayer(
            d_model=self.item_feat.size,
            nhead=2,
            dim_feedforward=self.item_feat.size,
            dropout=self.trans_dropout,
            batch_first=True)
        self.dissatis_his_transformer_layer = nn.TransformerEncoder(
            self.transformerEncoderLayer_dissatis, num_layers=1)
        
        self.rec_pos_emb = PositionalEmbedding(const.max_rec_his_len, self.item_feat.size)
        self.satis_pos_emb = PositionalEmbedding(const.max_satis_his_len, self.item_feat.size)
        self.dissatis_pos_emb = PositionalEmbedding(const.max_dissatis_his_len, self.item_feat.size)
    
        self.drop_out=nn.Dropout(config['dropout'])

        self.satis_his_att_pooling = Target_Attention( 
                        self.item_feat.size,
                        self.item_feat.size
                        )
        self.dissatis_his_att_pooling = Target_Attention( 
                self.item_feat.size,
                self.item_feat.size
                )
        
        self.get_weight_dissatis = nn.Linear(self.item_feat.size, 1)
        self.get_weight_satis = nn.Linear(self.item_feat.size, 1)
        self.get_group = nn.Linear(self.item_feat.size, 1)
        
        self.user2item = nn.Linear(self.user_feat.size, self.item_feat.size)
        self.get_weight_3 = nn.Linear(self.item_feat.size, 1)
        pred_hid_units=[self.item_feat.size] 
        self.fc_layer_item = FullyConnectedLayer(input_size = self.item_feat.size,
                                    hidden_unit=pred_hid_units,
                                    batch_norm=False,
                                    sigmoid = False,
                                    activation='relu',
                                    dropout=self.mlp_dropout,
                                    )
        self.W2 = torch.nn.Parameter(data=torch.randn(self.item_feat.size, self.item_feat.size), requires_grad=True)
        self.sigmoid = nn.Sigmoid()
        
        self.loss_func = nn.BCELoss()
        self.loss_func_t = nn.BCELoss(reduction='none')

        self._init_weights()
        all_params = torch.load(pre_trained_model,map_location=device)

        desired_params = {}
        for name, param in all_params.items():
            desired_params[name] = param
        result=self.load_state_dict(desired_params, strict=False)

        
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
    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)
    def input_from_feature_tables(self, input_data):

        user, rec_his,satis_his,dissatis_his, pos_item, neg_items,satis  = input_data
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

        return user_emb, pos_item_emb,neg_item_emb, rec_his_emb, rec_his_mask,satis_his_emb,satis_his_mask,dissatis_his_emb,dissatis_his_mask,satis
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
    
    def get_his_satis_emb(self,user,rec_his,rec_his_emb,rec_his_mask,batch_size):
        rec_his_satis = [tensor.view(-1) for tensor in rec_his]

        user_his = [  torch.repeat_interleave(user_f, rec_his_emb.size(1), dim=0)
                        for user_f in user
                    ]
        input_data_satis=[]
        input_data_satis.append(user_his)
        input_data_satis.append(rec_his_satis)
        with torch.no_grad():
            his_satis_label=satis_model(input_data_satis).view(batch_size,-1)
            his_satis_label[rec_his_mask] = 0 

        rec_his_emb[rec_his_mask] = 0.0
        
        his_satis_label = his_satis_label.masked_fill(rec_his_mask,float('nan'))
        
        num_interest=self.num_interest
        score=his_satis_label
        qq=torch.arange(1/num_interest,1+1/num_interest,1/num_interest).to(device)
        quantile_value = torch.nanquantile(input=score,q=qq,dim=1)

        his_satis_label = his_satis_label.masked_fill(rec_his_mask,float('0'))
        score=his_satis_label
        final_interest_mask = torch.zeros((num_interest,batch_size,rec_his_emb.size(1)))

        for i in range(num_interest):
            if i == 0:
                final_interest_mask[i,:,:] =  score <= quantile_value[i].unsqueeze(1)
            else:
                final_interest_mask[i,:,:] = (score <= quantile_value[i].unsqueeze(1)) & (score > quantile_value[i-1].unsqueeze(1))
                
        is_all_zero_mask = ~(final_interest_mask.sum(dim=2) == 0).to(device)
        
        score_expanded = score.unsqueeze(1)  
        mask_expanded = final_interest_mask.unsqueeze(2).to(device)   

        masked_score = score_expanded * mask_expanded
        masked_score = torch.where(masked_score == 0, torch.tensor(-1e14, dtype=masked_score.dtype).to(device), masked_score)
        
        masked_score=torch.softmax(masked_score,dim=-1)  
        
        muti_interests = torch.matmul(masked_score, torch.matmul(rec_his_emb, self.W2))

        
        groups_vector=muti_interests.transpose(0,1) .squeeze(2)
        groups_vector1=self.get_group(groups_vector).squeeze()
    
        groups_vector1=is_all_zero_mask.transpose(0,1)*groups_vector1
        groups_vector1 = torch.where(groups_vector1 == 0, torch.tensor(-1e14, dtype=groups_vector1.dtype).to(device), groups_vector1)
        
        group_attention_weights = F.softmax(groups_vector1, dim=-1)
        group_attention_weights = group_attention_weights.view(group_attention_weights.size(0), -1, 1) 

        final_result = torch.sum(groups_vector * group_attention_weights, dim=1)
        return final_result

    def forward(self, input_data,epoch,labels=None):
        """
        :param 
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)``
        numerical_x: Long tensor of size ``(batch_size, numerical_num)``
        """
    
        user_emb, pos_item_emb,neg_item_emb, rec_his_emb, rec_his_mask,satis_his_emb,satis_his_mask,dissatis_his_emb,dissatis_his_mask ,satis = self.input_from_feature_tables(input_data=input_data)
        
        batch_size=user_emb.size(0)
        input_lengths = torch.sum(~rec_his_mask, dim=1)
        satis_input_lengths = torch.sum(~satis_his_mask, dim=1)
        dissatis_input_lengths = torch.sum(~dissatis_his_mask, dim=1)
        user, rec_his,satis_his,dissatis_his, pos_item, neg_items,satis = input_data
        
        rec_his_emb += self.rec_pos_emb(rec_his_emb)
        satis_his_emb += self.satis_pos_emb(satis_his_emb)
        dissatis_his_emb += self.dissatis_pos_emb(dissatis_his_emb)
        
        rec_his_emb=self.drop_out(rec_his_emb)
        satis_his_emb=self.drop_out(satis_his_emb)
        dissatis_his_emb=self.drop_out(dissatis_his_emb)
        user, rec_his,satis_his,dissatis_his, pos_item, neg_items,satis = input_data
        
        rec_his_emb = self.rec_his_transformer_layer(src=rec_his_emb, src_key_padding_mask=rec_his_mask) # batch, max_len, dim
        satis_his_emb = self.satis_his_transformer_layer(src=satis_his_emb, src_key_padding_mask=satis_his_mask) # batch, max_len, dim
        dissatis_his_emb = self.dissatis_his_transformer_layer(src=dissatis_his_emb, src_key_padding_mask=dissatis_his_mask) # batch, max_len, dim

        result_tensor=self.get_his_satis_emb(user,rec_his,rec_his_emb,rec_his_mask,batch_size)

        rec_his_vector = self.gather_indexes(rec_his_emb, input_lengths - 1)
        satis_vector = self.gather_indexes(satis_his_emb, satis_input_lengths - 1)
        dissatis_vector = self.gather_indexes(dissatis_his_emb, dissatis_input_lengths - 1)
        
        rec_his_satis_emb = self.satis_his_att_pooling(rec_his_emb, satis_vector, rec_his_mask)
        rec_his_dissatis_emb = self.dissatis_his_att_pooling(rec_his_emb, dissatis_vector, rec_his_mask)
        
        satis_vector = torch.cat([vec.unsqueeze(1) for vec in [rec_his_satis_emb, satis_vector]], dim=1) #user大小不一样 b,d
        
        s_attention_weights = F.softmax(self.get_weight_satis(satis_vector).squeeze(), dim=-1) 
        
        s_attention_weights = s_attention_weights.view(s_attention_weights.size(0), -1, 1) 
        
        satis_his_cat = torch.sum(satis_vector * s_attention_weights, dim=1)
        
        dissatis_vector = torch.cat([vec.unsqueeze(1) for vec in [rec_his_dissatis_emb, dissatis_vector]], dim=1) #user大小不一样 #不用user_emb，效果好
        diss_attention_weights = F.softmax(self.get_weight_dissatis(dissatis_vector).squeeze(), dim=-1)
        diss_attention_weights = diss_attention_weights.view(diss_attention_weights.size(0), -1, 1) 
        dissatis_his_cat = torch.sum(dissatis_vector * diss_attention_weights, dim=1)
        
        user_emb=self.user2item(user_emb)

        allvec = torch.cat([vec.unsqueeze(1) for vec in [satis_his_cat, dissatis_his_cat, rec_his_vector,result_tensor,user_emb]], dim=1) 
        
        attention_weights = F.softmax(self.get_weight_3(allvec).squeeze(), dim=-1)

        attention_weights = attention_weights.view(attention_weights.size(0), -1, 1) 

        uemb = torch.sum(allvec * attention_weights, dim=1)  

        user_feats = [uemb]
        repeat_user_feats = [  torch.repeat_interleave(user_f, neg_item_emb.size(1), dim=0)
                                for user_f in user_feats
                            ] 
        
        pos_logits = self.sigmoid(
            (uemb * pos_item_emb).sum(-1)).view(batch_size, -1)
        neg_logits = self.sigmoid(
            (repeat_user_feats[0] * neg_item_emb.reshape(-1, neg_item_emb.size(-1))).sum(-1)).view(batch_size, -1)
        
        satis_pos_item_emb=self.fc_layer_item(pos_item_emb)
        satis_neg_item_emb=self.fc_layer_item(neg_item_emb.reshape(-1, neg_item_emb.size(-1)))
        
        pos_satis = self.sigmoid(
            (uemb * satis_pos_item_emb).sum(-1)).view(batch_size, -1)

        neg_satis = self.sigmoid(
            (repeat_user_feats[0] * satis_neg_item_emb).sum(-1)).view(batch_size, -1)
        
        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        labels = torch.zeros_like(logits).to(device)
        labels[:, 0] = 1.0

        logits = logits.reshape((-1,))
        labels = labels.reshape((-1,))

        loss = self.loss_func(logits, labels)

        # # for satis tower
        user, rec_his,satis_his,dissatis_his, pos_item, neg_items,satis = input_data
        pos_item_satis=pos_item
        input_data_satis=[]
        input_data_satis.append(user)
        input_data_satis.append(pos_item_satis)
        with torch.no_grad():
            pos_satis_label=satis_model(input_data_satis)
        pos_satis_label_tea=pos_satis_label.view(-1)
        
        if epoch>=self.epoch_correct:
            p_st=pos_satis.view(-1).detach()
            p_slt=pos_satis_label_tea.detach()

            loss_t = self.loss_func_t(p_st, p_slt)
            loss_t = min_max_normalize(loss_t)
            wei=generate_weight(loss_t).to(device) 
            pos_satis_label_tea=wei*p_slt+(1-wei)*p_st
            pos_satis_label_tea=pos_satis_label_tea.float()
        pos_satis_label=pos_satis_label_tea.view(batch_size,-1)

        neg_item_satis=neg_items

        neg_item_satis = [tensor.view(-1) for tensor in neg_item_satis]

        user2 = [  torch.repeat_interleave(user_f, neg_item_emb.size(1), dim=0)
                        for user_f in user
                    ]

        input_data_satis=[]
        input_data_satis.append(user2)
        input_data_satis.append(neg_item_satis)
        
        with torch.no_grad():
            neg_satis_label=satis_model(input_data_satis)
            
        
        if epoch>=self.epoch_correct:
            n_st=neg_satis.view(-1).detach()
            n_slt=neg_satis_label.view(-1).detach()
            
            loss_t = self.loss_func_t(n_st, n_slt)
            loss_t = min_max_normalize(loss_t)

            wei=generate_weight(loss_t).to(device)
            neg_satis_label=wei*n_slt+(1-wei)*n_st
            
            neg_satis_label=neg_satis_label.float()
        
        neg_satis_label=neg_satis_label.view(batch_size,-1)
    
        satis_preds = torch.cat([pos_satis, neg_satis], dim=-1)
        satis_labels = torch.cat([pos_satis_label, neg_satis_label], dim=-1)

        satis_preds = satis_preds.reshape((-1,))
        satis_labels = satis_labels.reshape((-1,))    
        loss2 = self.loss_func(satis_preds, satis_labels)
        return loss+self.weight_2*loss2
    
    
    def predict(self, input_data,labels=None):
        """
        :param 
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)``
        numerical_x: Long tensor of size ``(batch_size, numerical_num)``
        """
        user_emb, pos_item_emb, rec_his_emb, rec_his_mask,satis_his_emb,satis_his_mask,dissatis_his_emb,dissatis_his_mask  = self.input_from_feature_tables_test(input_data=input_data)
        
        batch_size=user_emb.size(0)
        input_lengths = torch.sum(~rec_his_mask, dim=1)
        satis_input_lengths = torch.sum(~satis_his_mask, dim=1)
        dissatis_input_lengths = torch.sum(~dissatis_his_mask, dim=1)
        
        rec_his_emb += self.rec_pos_emb(rec_his_emb)
        satis_his_emb += self.satis_pos_emb(satis_his_emb)
        dissatis_his_emb += self.dissatis_pos_emb(dissatis_his_emb)
        
        user, rec_his,satis_his,dissatis_his, pos_item = input_data
        rec_his_emb = self.rec_his_transformer_layer(src=rec_his_emb, src_key_padding_mask=rec_his_mask) # batch, max_len, dim
        satis_his_emb = self.satis_his_transformer_layer(src=satis_his_emb, src_key_padding_mask=satis_his_mask) # batch, max_len, dim
        dissatis_his_emb = self.dissatis_his_transformer_layer(src=dissatis_his_emb, src_key_padding_mask=dissatis_his_mask) # batch, max_len, dim
        
        result_tensor=self.get_his_satis_emb(user,rec_his,rec_his_emb,rec_his_mask,batch_size)
        

        rec_his_vector = self.gather_indexes(rec_his_emb, input_lengths - 1)
        satis_vector = self.gather_indexes(satis_his_emb, satis_input_lengths - 1) 
        dissatis_vector = self.gather_indexes(dissatis_his_emb, dissatis_input_lengths - 1)
        
        rec_his_satis_emb = self.satis_his_att_pooling(rec_his_emb, satis_vector, rec_his_mask)
        rec_his_dissatis_emb = self.dissatis_his_att_pooling(rec_his_emb, dissatis_vector, rec_his_mask)
        
        satis_vector = torch.cat([vec.unsqueeze(1) for vec in [rec_his_satis_emb, satis_vector]], dim=1)
        s_attention_weights = F.softmax(self.get_weight_satis(satis_vector).squeeze(), dim=-1)
        s_attention_weights = s_attention_weights.view(s_attention_weights.size(0), -1, 1) 
        satis_his_cat = torch.sum(satis_vector * s_attention_weights, dim=1)
        
        dissatis_vector = torch.cat([vec.unsqueeze(1) for vec in [rec_his_dissatis_emb, dissatis_vector]], dim=1) 
        diss_attention_weights = F.softmax(self.get_weight_dissatis(dissatis_vector).squeeze(), dim=-1)
        diss_attention_weights = diss_attention_weights.view(diss_attention_weights.size(0), -1, 1) 
        dissatis_his_cat = torch.sum(dissatis_vector * diss_attention_weights, dim=1)
        
        user_emb=self.user2item(user_emb)
        allvec = torch.cat([vec.unsqueeze(1) for vec in [satis_his_cat, dissatis_his_cat, rec_his_vector,result_tensor,user_emb]], dim=1) 
        attention_weights = F.softmax(self.get_weight_3(allvec).squeeze(), dim=-1)
        attention_weights = attention_weights.view(attention_weights.size(0), -1, 1) 
        uemb = torch.sum(allvec * attention_weights, dim=1)

        pos_logits = self.sigmoid(
            (uemb * pos_item_emb).sum(-1)).view(batch_size, -1)
        
        return pos_logits

class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, dim):
        super().__init__()

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, dim)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)

class Target_Attention(nn.Module):
    def __init__(self, hid_dim1, hid_dim2):
        super().__init__()
        
        self.W = nn.Parameter(torch.randn((1, hid_dim1, hid_dim2)))
        nn.init.xavier_normal_(self.W)
        self.softmax = nn.Softmax(dim=-1)

     
    def forward(self, seq_emb, target, mask):
        '''
        Args:
            seq_emb: batch, seq_length, dim1
            target: batch, dim2
            mask: batch, seq_length. True means padding
            pos_mask: batch, seq_length. True means postive elements
            neg_mask: batch, seq_length. True means negative elements
        '''

        score = torch.matmul(seq_emb, self.W) #batch, seq, dim2
        score = torch.matmul(score, target.unsqueeze(-1)) #batch, seq, 1
        
        all_score = score.masked_fill(mask.unsqueeze(-1), torch.tensor(-1e16))
        all_weight = self.softmax(all_score.transpose(-2, -1)) #batch,1,seq
        all_vec = torch.matmul(all_weight, seq_emb).squeeze(1) #batch, dim1

        return all_vec

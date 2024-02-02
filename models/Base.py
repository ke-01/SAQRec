import torch
import torch.nn as nn

from .module import FullyConnectedLayer,SharedBottomModel2
from .Inputs import user_feat, item_feat
from config import const
import torch.nn.functional as F
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
device = torch.device("cuda")

class Base(torch.nn.Module):
    """
    A pytorch implementation of Base Model.
    """

    def __init__(self, config):
        super().__init__()
        
        self.user_feat = user_feat()
        self.item_feat = item_feat()
        
        self.transformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.item_feat.size,
            nhead=2,
            dim_feedforward=self.item_feat.size,
            dropout=config['trans_dropout'],
            batch_first=True)
        self.rec_his_transformer_layer = nn.TransformerEncoder(
            self.transformerEncoderLayer, num_layers=1)
 
        self.rec_pos_emb = PositionalEmbedding(const.max_rec_his_len, self.item_feat.size)
        self.user2item = nn.Linear(self.user_feat.size, self.item_feat.size)
        self.get_weight_3 = nn.Linear(self.item_feat.size, 1)

        self.sigmoid = nn.Sigmoid()
        self.loss_func = nn.BCELoss()

        self._init_weights()
        
        
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
        """
        :param 
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)``
        numerical_x: Long tensor of size ``(batch_size, numerical_num)``
        """
    
        user_emb, pos_item_emb,neg_item_emb, rec_his_emb, rec_his_mask,satis_his_emb,satis_his_mask,dissatis_his_emb,dissatis_his_mask  = self.input_from_feature_tables(input_data=input_data)
        
        batch_size=user_emb.size(0)
        input_lengths = torch.sum(~rec_his_mask, dim=1)
        
        rec_his_emb +=self.rec_pos_emb(rec_his_emb)
        rec_his_emb = self.rec_his_transformer_layer(src=rec_his_emb, src_key_padding_mask=rec_his_mask) # batch, max_len, dim
        rec_his_vector = self.gather_indexes(rec_his_emb, input_lengths - 1)

        user_emb=self.user2item(user_emb)
        allvec = torch.cat([vec.unsqueeze(1) for vec in [ rec_his_vector,user_emb]], dim=1) 
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

        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        labels = torch.zeros_like(logits).to(device)
        labels[:, 0] = 1.0

        logits = logits.reshape((-1,))
        labels = labels.reshape((-1,))

        loss = self.loss_func(logits, labels)
                
        return loss
    def predict(self, input_data,labels=None):

        
        user_emb, pos_item_emb, rec_his_emb, rec_his_mask,satis_his_emb,satis_his_mask,dissatis_his_emb,dissatis_his_mask  = self.input_from_feature_tables_test(input_data=input_data)

        batch_size=user_emb.size(0)
        input_lengths = torch.sum(~rec_his_mask, dim=1)
 
        rec_his_emb += self.rec_pos_emb(rec_his_emb)
 
        rec_his_emb = self.rec_his_transformer_layer(src=rec_his_emb, src_key_padding_mask=rec_his_mask) # batch, max_len, dim
        rec_his_vector = self.gather_indexes(rec_his_emb, input_lengths - 1)

        user_emb=self.user2item(user_emb)
        allvec = torch.cat([vec.unsqueeze(1) for vec in [ rec_his_vector,user_emb]], dim=1) 

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
    
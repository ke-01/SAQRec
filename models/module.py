
import torch.nn as nn
import torch
from .Inputs import user_feat, item_feat
from config import const
import torch.nn.functional as F
from scipy.stats import beta

class FullyConnectedLayer(nn.Module):
    def __init__(self, input_size, hidden_unit, batch_norm=False, activation='relu', sigmoid=False, dropout=None, dice_dim=None):
        super(FullyConnectedLayer, self).__init__()
        assert len(hidden_unit) >= 1 
        self.sigmoid = sigmoid

        layers = []
        layers.append(nn.Linear(input_size, hidden_unit[0]))
        
        for i, h in enumerate(hidden_unit[:-1]):
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_unit[i]))
            
            if activation.lower() == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation.lower() == 'tanh':
                layers.append(nn.Tanh())
            elif activation.lower() == 'leakyrelu':
                layers.append(nn.LeakyReLU())
            else:
                raise NotImplementedError

            if dropout is not None:
                layers.append(nn.Dropout(dropout))
            
            layers.append(nn.Linear(hidden_unit[i], hidden_unit[i+1]))
        
        self.fc = nn.Sequential(*layers)
        if self.sigmoid:
            self.output_layer = nn.Sigmoid()
        

    def forward(self, x):
        return self.output_layer(self.fc(x)) if self.sigmoid else self.fc(x) 

class Satis_Model(torch.nn.Module):
    def __init__(self, clamp,drop_out):
        super().__init__()
        
        self.user_feat = user_feat()
        self.item_feat = item_feat()

        self.drop_out=drop_out
        self.clamp=clamp
        self.embed_output_dim=self.item_feat.size+self.user_feat.size
        

        pred_hid_units=[200, 80, 1]

        self.fc_layer = FullyConnectedLayer(input_size = self.embed_output_dim,
                    hidden_unit=pred_hid_units,
                    batch_norm=False,
                    sigmoid = True,
                    activation='relu',
                    dropout=self.drop_out,
                    )
        
        self.loss_func = nn.BCELoss(reduction='none')
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


    def input_from_feature_tables(self, input_data):
        #batch, feature

        user, item= input_data


        user_emb = self.user_feat.get_emb(user)
        item_emb = self.item_feat.get_emb(item)
        
        
        return user_emb, item_emb

    def forward(self, input_data,labels=None):
        
        user_emb, item_emb = self.input_from_feature_tables(input_data=input_data)
        emb = torch.cat([item_emb,  user_emb], dim=1)
        logits=(self.fc_layer(emb).squeeze(1))
        if labels is not None:            
            return 0
        else:
            return logits

def min_max_normalize(tensor, min_value=0, max_value=1, epsilon=1e-8):
    min_val = tensor.min()
    max_val = tensor.max()

    denominator = max_val - min_val
    denominator = torch.where(denominator > epsilon, denominator, torch.tensor(epsilon, dtype=tensor.dtype).to(device))

    normalized_tensor = min_value + (max_value - min_value) * (tensor - min_val) / denominator
    return normalized_tensor

def generate_weight(x):
    x_numpy = x.cpu().numpy() 
    weight = beta.cdf(x_numpy, 0.5, 0.5)
    return torch.tensor(weight)  



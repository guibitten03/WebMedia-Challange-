import torch
import torch.nn as nn
import time

from .prediction import PredictionLayer
from .fusion import FusionLayer


class Model(nn.Module):
    
    def __init__(self, config, net, model_name):
        super(Model, self).__init__()

        self.config = config
        self.model_name = model_name
        self.net = net


        if self.config['ui_merge'] == 'cat':
            if self.config['r_id_merge'] == 'cat':
                    config['feature_dim'] = self.config['id_emb_size'] * self.config['num_fea'] * 2
            else:
                    config['feature_dim'] = self.config['id_emb_size'] * 2
        else:
            if self.config['r_id_merge'] == 'cat':
                    config['feature_dim'] = self.config['id_emb_size'] * self.config['num_fea']
            else:
                    config['feature_dim'] = self.config['id_emb_size']

        self.config['feature_dim'] = config['feature_dim']

        self.fusion_net = FusionLayer(config)
        self.predict_layer = PredictionLayer(config)
        self.dropout = nn.Dropout(config['drop_out'])


    def forward(self, datas):
        uids, idds, user_doc, item_doc = datas

        u_fea, i_fea = self.net(datas)
        ui_fea = self.fusion_net([u_fea, i_fea])
        ui_fea = self.dropout(ui_fea)
        output = self.predict_layer(ui_fea, uids, idds).squeeze(1)
        return output
    
    
    def load(self, path):
        
        self.load_state_dict(torch.load(path))
        

    def save(self, epoch=None, name=None, opt=None):
        
        prefix = 'checkpoints/'
        if name is None:
            name = prefix + self.model_name + '_'
            name = time.strftime(name + '%m%d_%H:%M:%S.pth')
        else:
            name = prefix + self.model_name + '.pth'
        torch.save(self.state_dict(), name)
        return name
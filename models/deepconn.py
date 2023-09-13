import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class DeepCoNN(nn.Module):
    def __init__(self, general_config):
        super(DeepCoNN, self).__init__()
        self.config = general_config

        # Transformação dos vetores já criou as emdeddings
        # self.user_word_embs = nn.Embedding(general_config['vocab_size'], general_config['word_dim'])  # vocab_size * 300
        # self.item_word_embs = nn.Embedding(general_config['vocab_size'], general_config['word_dim'])  # vocab_size * 300

        self.conv_u = torch.nn.Conv2d(
            1, general_config['filters_num'], 
            kernel_size=(general_config['kernel_size'], general_config['word_dim']))
        
        self.conv_i = torch.nn.Conv2d(
            1, general_config['filters_num'], 
            kernel_size=(general_config['kernel_size'], general_config['word_dim']))

        self.u_linear = torch.nn.Linear(general_config['filters_num'], general_config['fc_dim'])
        self.i_linear = torch.nn.Linear(general_config['filters_num'], general_config['fc_dim'])
        
        self.dropout = torch.nn.Dropout(0.5)

        self.reset_para()

    
    def forward(self, datas):
        uids, iids, user_doc, item_doc = datas

        # user_doc = self.user_word_embs(user_doc)
        # item_doc = self.item_word_embs(item_doc)

        if user_doc.shape[0] == 768:
            user_doc = user_doc.T
        if item_doc.shape[0] == 768:
            item_doc = item_doc.T


        u_fea = F.relu(self.conv_u(user_doc.unsqueeze(0).unsqueeze(3))).squeeze(3)
        i_fea = F.relu(self.conv_i(item_doc.unsqueeze(0).unsqueeze(3))).squeeze(3)

        u_fea = F.max_pool1d(u_fea, u_fea.size(2)).squeeze(2)
        i_fea = F.max_pool1d(i_fea, i_fea.size(2)).squeeze(2)

        u_fea = self.dropout(self.u_linear(u_fea))
        i_fea = self.dropout(self.i_linear(i_fea))

        out = torch.stack([u_fea], 1), torch.stack([i_fea], 1)

        return out
        # return self.fusion_layer(out, 0, 0)

        
    def reset_para(self):
        for cnn in [self.conv_u, self.conv_i]:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)

        for fc in [self.u_linear, self.i_linear]:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.constant_(fc.bias, 0.1)

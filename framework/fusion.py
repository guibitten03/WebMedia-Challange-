import torch
import torch.nn as nn

class FusionLayer(nn.Module):

    def __init__(self, config):
        super(FusionLayer, self).__init__()

        self.config = config
        self.attn = SelfAtt(config['id_emb_size'], config['num_heads'])

        
    def forward(self, ui_features):
        out = self.attn(ui_features[0], ui_features[1])
        s_u_out, s_i_out = torch.split(out, out.size(1)//2, 1)

        u_out = ui_features[0] + s_u_out
        i_out = ui_features[1] + s_i_out

        u_out = u_out.reshape(u_out.size(0), -1)
        i_out = i_out.reshape(u_out.size(0), -1)

        out = torch.cat([u_out, i_out], 1)

        return out



class SelfAtt(nn.Module):
    '''
    self attention for interaction
    '''
    def __init__(self, dim, num_heads):
        super(SelfAtt, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(dim, num_heads, 128, 0.4)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, 1)

    def forward(self, user_fea, item_fea):
        fea = torch.cat([user_fea, item_fea], 1).permute(1, 0, 2)  # batch * 6 * 64
        out = self.encoder(fea)
        return out.permute(1, 0, 2)
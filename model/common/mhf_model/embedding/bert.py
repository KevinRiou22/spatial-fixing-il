import torch.nn as nn
import torch
from .token import TokenEmbedding
from .position import PositionalEmbedding


class BERTEmbedding(nn.Module):
    def __init__(self,cfg, inp_channels,  embed_size, dropout=0.1, max_len = 7, istrain=False):
        super().__init__()
        self.cfg = cfg
        self.token = TokenEmbedding(cfg, inp_channels = inp_channels, embed_size=embed_size)
        self.position = PositionalEmbedding(cfg, d_model=embed_size, max_len = max_len)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size
        if istrain:  
            self.view_num = len(cfg.H36M_DATA.TRAIN_CAMERAS) + cfg.TRAIN.NUM_AUGMENT_VIEWS
        else:
            self.view_num = len(cfg.H36M_DATA.TRAIN_CAMERAS)
        self.view_embed = nn.Parameter(torch.zeros(len(cfg.H36M_DATA.TRAIN_CAMERAS) + cfg.TRAIN.NUM_AUGMENT_VIEWS, 1, embed_size))

    def forward(self, sequence):
        position, mask = self.position(sequence)
        token = self.token(sequence)
        token = token.view(-1, self.view_num, token.shape[-2], token.shape[-1]).contiguous()
        x = token + position.unsqueeze(0) + self.view_embed[:token.shape[1]].unsqueeze(0).repeat([token.shape[0], 1, token.shape[-2], 1])
        x = x.view(-1, token.shape[-2], token.shape[-1]).contiguous()
        return self.dropout(x), mask
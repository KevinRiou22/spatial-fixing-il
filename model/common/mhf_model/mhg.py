import math
import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath
from .transformer import TransformerBlock
from .embedding import BERTEmbedding
from .utils.gelu import *

class mhg(nn.Module):
    def __init__(self, cfg, in_channels, hidden, num_heads, dropout, T=7, is_last=False, istrain=False):
        super(mhg, self).__init__()
        self.cfg = cfg
        self.in_channels = cfg.NETWORK.NUM_CHANNELS
        self.hidden = cfg.NETWORK.T_FORMER.NUM_CHANNELS
        self.num_heads = num_heads
        self.dropout = dropout
        self.n_layers = cfg.NETWORK.T_FORMER.NUM_LAYERS
        #
        self.embedding = BERTEmbedding(cfg, inp_channels=self.in_channels, embed_size=self.hidden, max_len=T, istrain=istrain)
        #
        self.Transformer_block = nn.ModuleList([
            TransformerBlock(cfg, self.hidden, self.num_heads, dropout=self.dropout, feed_forward_hidden=self.hidden*2, T=T, is_last = i == self.n_layers - 1)
            for i in range(self.n_layers)])
    def set_bn_momentum(self, momentum):
        for t in self.Transformer_block:
            t.set_bn_momentum(momentum)
    def forward(self, x):

        inp, mask = self.embedding(x)
        x_0 = inp + self.Transformer_block[0].forward(inp, mask)
        x_1 = x_0 + self.Transformer_block[1].forward(x_0, mask)
        x_2 = x_1 + self.Transformer_block[2].forward(x_1, mask)
        x_3 = x_2 + self.Transformer_block[3].forward(x_2, mask)

        return x_0, x_1, x_2, x_3

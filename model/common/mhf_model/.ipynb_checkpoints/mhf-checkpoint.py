import torch.nn as nn
import torch
from .transformer import TransformerBlock
from .embedding import BERTEmbedding
from .utils.gelu import *
from .mhg import mhg
from .trans_blk_frm_mht import Transformer as TransformerBlock_frm_mhg
from .trans_hypothesis import Transformer as Transformer_hypothesis


class Head(nn.Module):
    def __init__(self, cfg, in_channels=10, hidden=512, dropout=0.25, channels=2048, num_joints=17, istrain=False):
        super().__init__()
        self.cfg = cfg
        channels = in_channels
        self.hidden=hidden
        self.frame_len = cfg.NETWORK.TEMPORAL_LENGTH
        self.transf_dim= cfg.NETWORK.TRANSFORM_DIM
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        if istrain:
            self.view_num = len(cfg.H36M_DATA.TRAIN_CAMERAS) + cfg.TRAIN.NUM_AUGMENT_VIEWS
        else:
            self.view_num = len(cfg.H36M_DATA.TRAIN_CAMERAS)
        self.shrink = nn.Conv1d(channels, self.hidden, 1, bias=True)

        self.num_joints = num_joints

    def set_bn_momentum(self, momentum):
        pass

    def forward(self, x):
        B, T, C = x.shape

        #x = x[:, T // 2:(T // 2 + 1)]
        x = x.permute(0, 2, 1).contiguous()
        x = self.shrink(x).view(-1, self.view_num, self.hidden//self.transf_dim, self.transf_dim, self.frame_len).permute(0, 2, 3, 4, 1).contiguous()
        
        return x


class MyConv(nn.Module):
    def __init__(self, cfg, V, channels):
        super().__init__()

        self.expand_conv = nn.Conv2d(V, V * cfg.NETWORK.INPUT_DIM * channels, (1, 1), stride=(1, 1), bias=False)

    def forward(self, pos_2d, vis_score):
        conv_p = self.expand_conv(vis_score)  # (B, C_1*C_2, T, N)
        B, _, T, N = conv_p.shape

        conv_p = conv_p.view(B, pos_2d.shape[1], -1, T, N)
        x = torch.einsum('bcktn, bctn -> bktn', conv_p, pos_2d).contiguous()
        return x

class pre_proj(nn.Module):
    def __init__(self, cfg, in_N, h_N, dropout = 0.25, momentum = 0.1, is_train = False, CONF_MOD=True):
        super(pre_proj, self).__init__()
        self.cfg = cfg
        if cfg.NETWORK.CONFIDENCE_METHOD == 'no':
            self.CAT_CONF = False
            self.CONF_MOD = False
        elif cfg.NETWORK.CONFIDENCE_METHOD == 'concat':
            self.CAT_CONF = True
            self.CONF_MOD = False
        elif cfg.NETWORK.CONFIDENCE_METHOD == 'modulate':
            self.CAT_CONF = False
            self.CONF_MOD = True
        cord_D = self.cfg.NETWORK.INPUT_DIM
        in_channels = in_N * cord_D
        h_D = cfg.NETWORK.TRANSFORM_DIM  #4
        channels = h_N * h_D

        self.expand_conv = nn.ModuleList([
            nn.Conv2d(in_channels, channels, (1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(channels, momentum=momentum),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        ])
        self.num_layers = 2
        conv_layers = []
        bn_layers = []
        for i in range(self.num_layers):
            conv_layers.append(nn.Conv2d(channels, channels, 1, bias=False))
            bn_layers.append(nn.BatchNorm2d(channels, momentum=momentum))
            conv_layers.append(nn.Conv2d(channels, channels, 1, bias=False))
            bn_layers.append(nn.BatchNorm2d(channels, momentum=momentum))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        if self.CONF_MOD:
            self.vis_conv = MyConv(cfg, V=in_N, channels=h_N * cfg.NETWORK.TRANSFORM_DIM)
    def set_bn_momentum(self, momentum):
        self.expand_conv[1].momentum = momentum
        for bn in self.bn_layers:
            bn.momentum = momentum

    def forward(self, pos_2d):

        vis_score = pos_2d[:, :, :, -1:]  # (B, T, J, C, N)
        pos_2d = pos_2d[:, :, :, :-1]

        B, T, V1, C1, N = pos_2d.shape

        pos_2d = pos_2d.permute(0, 2, 3, 1, 4).contiguous()  # (B, J, C, T, N)
        pos_2d = pos_2d.view(B, V1 * C1, T, N).contiguous()

        vis_score = vis_score.permute(0, 2, 3, 1, 4).contiguous()
        vis_score = vis_score.view(B, V1, T, N).contiguous()

        if self.CONF_MOD:
            vis_x = self.vis_conv(pos_2d, vis_score)

        if not self.CAT_CONF:
            x = pos_2d
        else:
            x = torch.cat((pos_2d, vis_score), dim=1)
        for m in self.expand_conv:
            x = m(x)
        if self.CONF_MOD:
            x = x + vis_x
        x = x.contiguous()
        K = 2
        for i in range(self.num_layers):
            res = x
            x = self.drop(self.relu(self.bn_layers[K * i](self.conv_layers[K * i](x))))
            x = self.drop(self.relu(self.bn_layers[K * i + 1](self.conv_layers[K * i + 1](x))))
            x = res + x

        return x.view(x.shape[0], -1, self.cfg.NETWORK.TRANSFORM_DIM, x.shape[-2], x.shape[-1])


class MHF(nn.Module):
    def __init__(self, cfg, in_channels, emb_size, T=7, dropout=0.1, num_joints=17, momentum = 0.1, istrain = False):
        super().__init__()
        self.cfg = cfg
        self.sub_type = cfg.NETWORK.SUB_TYPE
        self.in_channels = in_channels
        self.hidden = emb_size
        self.n_layers = cfg.NETWORK.T_FORMER.NUM_LAYERS
        self.attn_heads = cfg.NETWORK.T_FORMER.NUM_HEADS
        self.max_view_num = len(cfg.H36M_DATA.TRAIN_CAMERAS) + cfg.TRAIN.NUM_AUGMENT_VIEWS
        if istrain:
            self.view_num = len(cfg.H36M_DATA.TRAIN_CAMERAS) + cfg.TRAIN.NUM_AUGMENT_VIEWS
        else:
            self.view_num = len(cfg.H36M_DATA.TRAIN_CAMERAS)
        self.num_joints = num_joints
        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = self.hidden * 2
        self.T = T

        # Multi Hypothesis Generator
        self.mhg = mhg(cfg, in_channels=self.in_channels, hidden=self.hidden, num_heads=self.attn_heads, dropout=dropout, T = T, istrain = istrain)

        # supplement more view diversification by adding embedding
        self.view_embedding = nn.Parameter(torch.zeros(1, 1, 4, 1))\
            .repeat(self.cfg.TRAIN.BATCH_SIZE*(self.max_view_num), T, 1, self.hidden)
        # intermidiate shrink
        self.inter_shrink = nn.Linear(self.hidden*len(cfg.H36M_DATA.TRAIN_CAMERAS), self.hidden*3)
        #
        self.Transformer_hypothesis = Transformer_hypothesis(depth=3, embed_dim=self.hidden, mlp_hidden_dim=self.hidden*2, length=T)

        self.shrink = Head(cfg, in_channels=self.hidden*3, hidden=self.hidden*2, num_joints=num_joints, istrain=istrain)

    def set_bn_momentum(self, momentum):
        self.shrink.set_bn_momentum(momentum)
        for t in self.mhg:
            t.set_bn_momentum(momentum)

    def forward(self, x, rotation=None):
        if len(x.shape) == 5:
            B, C1, C2, T, N = x.shape
            x = x.view(B, -1, T, N)
        B, C, T, N = x.shape  # (B, C, T, N)

        x = x.permute(0, 3, 2, 1).contiguous()
        x = x.view(B * N, T, -1)

        B, T, C = x.shape

        inp = x

        # project hidden_dim from joints_num to hidden/2 (128 or 256 ?)

        # embedding the indexed sequence to sequence of vectors
        x_0, x_1, x_2, x_3 = self.mhg(inp)
        print('The input size is {}'.format(inp.shape))
        if self.sub_type=='views_augment':
        # add each view's feature into x_0, x_1, x_2, x_3 independently
            x_0 = x_0 + x.view(-1, N, T, self.hidden)[:, 0:1].squeeze(1).repeat([N, 1, 1])
            x_1 = x_1 + x.view(-1, N, T, self.hidden)[:, 1:2].squeeze(1).repeat([N, 1, 1])
            x_2 = x_2 + x.view(-1, N, T, self.hidden)[:, 2:3].squeeze(1).repeat([N, 1, 1])
            x_3 = x_3 + x.view(-1, N, T, self.hidden)[:, 3:4].squeeze(1).repeat([N, 1, 1])
        x = torch.cat([x_0, x_1, x_2, x_3], dim=-1)
        
        x += self.view_embedding[:B].view(x.shape[0], self.T, -1).to(x.device)
        # feature projection, aims to concatenate all 4 features and project them to 3 features corresponding to q, k, v
        
        x = self.inter_shrink(x)
        
        x = self.Transformer_hypothesis(x[:,:,:self.hidden], x[:, :, self.hidden:2*self.hidden], x[:,:,2*self.hidden:])

        x = self.shrink(x)
        #B, C1, C2, T, N = x.shape
        return x

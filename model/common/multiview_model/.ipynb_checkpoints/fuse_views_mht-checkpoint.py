from pydoc import visiblename
import torch.nn as nn
import torch
import numpy as np
import sys, os
import copy
import matplotlib.pyplot as plt
import torch.nn.functional as F
import itertools
import pickle
from ..mhf_model.mhf import *

from ..set_seed import *
from ..bert_model.bert import *
from .point_transformer_pytorch import *
from .video_multi_view import *
set_seed()


class Pose3dShrink_out(Pose3dShrinkOther):
        def __init__(self, cfg, N, channels, dropout = 0.25, momentum = 0.1, dim_joint = 3,is_train = False, num_joints = 17):
            super().__init__(cfg, N, channels, dropout, momentum, dim_joint, is_train, num_joints)
            self.cfg = cfg
            self.N = N
            self.channels = channels
            self.dropout = dropout
            self.dim_joint = dim_joint
            self.num_joints = num_joints
            self.training = is_train

            self.head = Head(cfg=self.cfg, in_channels=self.channels, dropout=self.dropout, num_joints=self.num_joints)

        def forward(self, x):
            B, _, _, T, N = x.shape
            x = x.view(x.shape[0], -1, T, N)
            x = self.drop(self.relu(self.bn_1(self.conv_1(x))))
            K = 2
            for i in range(self.num_layers):
                if self.training:
                    res = x[:, :, 1::3]
                else:
                    res = x[:, :, 3 ** (i + 1):-3 ** (i + 1)]
                x = self.drop(self.relu(self.bn_layers[K * i](self.conv_layers[K * i](x))))
                x = self.drop(self.relu(self.bn_layers[K * i + 1](self.conv_layers[K * i + 1](x))))
                x = res + x

            x = x.permute(0, 3, 2, 1).contiguous()
            x = x.view(B * N, T, -1)
                
            x = self.head(x)
            B, T, _, _ = x.shape
            x = x.view(-1, N, T, self.num_joints, 3)
            x = x.permute(0, 2, 3, 4, 1) #(B, T, J, 3, N)
            return x

class fuse_views_mht(nn.Module):
    def __init__(self, cfg, dropout=0.1, momentum=0.1, is_train=False, num_joints=17, ):
        super().__init__()
        self.cfg = cfg
        self.in_channels = cfg.NETWORK.NUM_CHANNELS
        self.hidden = cfg.NETWORK.T_FORMER.NUM_CHANNELS
        self.n_layers = cfg.NETWORK.T_FORMER.NUM_LAYERS
        self.attn_heads = cfg.NETWORK.T_FORMER.NUM_HEADS
        self.num_joints = num_joints
        use_inter_loss = cfg.TRAIN.USE_INTER_LOSS
        assert cfg.NETWORK.CONFIDENCE_METHOD in ['no', 'concat', 'modulate']

        if is_train:
            print('dim: {}'.format(cfg.NETWORK.TRANSFORM_DIM))
        self.T = cfg.NETWORK.TEMPORAL_LENGTH
        N_before_mhf = cfg.NETWORK.NUM_CHANNELS // cfg.NETWORK.TRANSFORM_DIM
        N_after_mhf = cfg.NETWORK.AFTER_MHF_DIM // cfg.NETWORK.TRANSFORM_DIM
        # pre-embeding hidden_dim from num_joints*2 to hidden/2
        self.pre_embedding = pre_proj(cfg, in_N=num_joints, h_N=N_before_mhf, dropout=dropout, is_train=is_train)
        # proposed mhf
        self.mhf = MHF(cfg, in_channels=self.in_channels, emb_size=self.hidden, T=self.T, dropout=dropout, num_joints=num_joints, istrain=is_train)

        if self.cfg.NETWORK.USE_MFT:
            self.fuse_model = FuseView(cfg, N_after_mhf, dropout=cfg.NETWORK.DROPOUT, is_train=is_train, T=self.T,
                                       num_joints=num_joints)

        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(cfg.NETWORK.DROPOUT)
        if is_train and use_inter_loss:
            self.shrink_1 = Pose3dShrinkOther(cfg, N=N_after_mhf, channels=cfg.NETWORK.NUM_CHANNELS,
                                              dropout=cfg.NETWORK.DROPOUT, dim_joint=3, is_train=is_train,
                                              num_joints=num_joints)
        self.shrink_out = Pose3dShrink_out(cfg=cfg, N=N_after_mhf, channels=cfg.NETWORK.NUM_CHANNELS,
                                                  dropout=cfg.NETWORK.DROPOUT, dim_joint=3, is_train=is_train,
                                                  num_joints=num_joints)
        self.use_inter_loss = use_inter_loss

        if self.cfg.NETWORK.USE_FEATURE_TRAN and is_train and self.cfg.NETWORK.M_FORMER.MODE == 'mtf':
            if not self.cfg.NETWORK.USE_GT_TRANSFORM:
                if self.cfg.TRAIN.USE_ROT_LOSS and self.training:
                    self.tran_shrink = RotShrink(cfg, N=N, channels=cfg.NETWORK.NUM_CHANNELS,
                                                 dropout=cfg.NETWORK.DROPOUT, dim_joint=3, is_train=is_train,
                                                 num_joints=num_joints)
            else:
                if self.cfg.TRAIN.USE_ROT_LOSS and self.cfg.NETWORK.M_FORMER.GT_TRANSFORM_RES:
                    self.tran_shrink = RotShrink(cfg, N=N, channels=cfg.NETWORK.NUM_CHANNELS,
                                                 dropout=cfg.NETWORK.DROPOUT, dim_joint=3, is_train=is_train,
                                                 num_joints=num_joints)

    def set_bn_momentum(self, momentum):
        ####fuse_model
        if self.cfg.NETWORK.USE_MFT:
            self.pre_embedding.set_bn_momentum(momentum)
            self.fuse_model.set_bn_momentum(momentum)
        ####shrink_1, shrink_out
        if self.training and self.use_inter_loss:
            self.shrink_1.set_bn_momentum(momentum)
        self.shrink_out.set_bn_momentum(momentum)
        ####tran_shrink
        if self.cfg.NETWORK.USE_FEATURE_TRAN and self.training and self.cfg.NETWORK.M_FORMER.MODE == 'mtf':
            if not self.cfg.NETWORK.USE_GT_TRANSFORM:
                if self.cfg.TRAIN.USE_ROT_LOSS and self.training:
                    self.tran_shrink.set_bn_momentum(momentum=momentum)
            else:
                if self.cfg.TRAIN.USE_ROT_LOSS and self.cfg.NETWORK.M_FORMER.GT_TRANSFORM_RES:
                    self.tran_shrink.set_bn_momentum(momentum=momentum)

    def forward(self, pos_2d, rotation=None):
        print('The size of pos_2d is {}'.format(pos_2d.shape))
        B, T, V, C, N = pos_2d.shape
        pos_2d = pos_2d.contiguous()
        print("_pos_2d.shape={}".format(pos_2d.shape))
        f = self.pre_embedding(pos_2d)
        f = self.mhf(f)  # (B, K, D, T, N)
        print("f_.shape={}".format(f.shape))
        if self.training and self.use_inter_loss:
            out_1 = self.shrink_1(f[:, :, :self.cfg.NETWORK.TRANSFORM_DIM].contiguous())
        tran = None
        rot = None
        f_fuse_before = f

        if self.cfg.NETWORK.USE_MFT:
            f, tran, att, tran_rot, f_tmp_rcpe, mask = self.fuse_model(f, pos_2d, rotation)
            if self.cfg.NETWORK.USE_FEATURE_TRAN and self.cfg.NETWORK.M_FORMER.MODE == 'mtf' and self.training:
                if not self.cfg.NETWORK.USE_GT_TRANSFORM:
                    if self.cfg.TRAIN.USE_ROT_LOSS:
                        rot = self.tran_shrink(tran)
                    else:
                        rot = None
                else:
                    if self.cfg.TRAIN.USE_ROT_LOSS and self.cfg.NETWORK.M_FORMER.GT_TRANSFORM_RES:
                        rot = self.tran_shrink(tran)
                    else:
                        rot = None
            else:
                rot = None
        out = self.shrink_out(f)

        if self.training and self.use_inter_loss:
            return out, [out_1, out_1] if self.cfg.NETWORK.USE_MFT else [out_1], tran, rot
        else:
            return out, [f_fuse_before]

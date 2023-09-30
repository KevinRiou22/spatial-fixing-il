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
from ..pose_resnet import PoseResNet
from torchvision.models import resnet18#resnet18#, ResNet18_Weights
#from torchvision.models.detection import keypointrcnn_resnet50_fpn#, KeypointRCNN_ResNet50_FPN_Weights
#from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchsummary import summary
from ..simple_HRNet.models_.hrnet import HRNet
set_seed()


class Pose3dShrink_out(Pose3dShrinkOther):
        def __init__(self, cfg, N, channels, dropout = 0.25, momentum = 0.1,is_train = False, num_joints = 17):
            super().__init__(cfg, N, channels, dropout, momentum, is_train, cfg.OBJ_POSE_DATA.NUM_OBJS )
            self.cfg = cfg
            self.N = N
            self.channels = channels
            self.dropout = dropout
            self.n_objs = cfg.OBJ_POSE_DATA.NUM_OBJS
            self.dim_obj = cfg.OBJ_POSE_DATA.DIM_OBJ
            self.n_wps = cfg.OBJ_POSE_DATA.NUM_WPS
            self.dim_wp = cfg.OBJ_POSE_DATA.DIM_WP
            self.training = is_train

            self.head = Head(cfg=self.cfg, in_channels=self.channels, dropout=self.dropout, num_joints=self.n_wps )

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
            x = x.view(-1, N, T, self.n_wps, self.dim_wp)
            x = x.permute(0, 2, 3, 4, 1) #(B, T, J, 3, N)
            return x

class fuse_views_mht(nn.Module):
    def __init__(self, cfg, dropout=0.1, momentum=0.1, is_train=False ):
        super().__init__()
        self.cfg = cfg
        self.in_channels = cfg.NETWORK.NUM_CHANNELS
        self.hidden = cfg.NETWORK.T_FORMER.NUM_CHANNELS
        self.n_layers = cfg.NETWORK.T_FORMER.NUM_LAYERS
        self.attn_heads = cfg.NETWORK.T_FORMER.NUM_HEADS
        self.n_objs = cfg.OBJ_POSE_DATA.NUM_OBJS
        self.dim_obj = cfg.OBJ_POSE_DATA.DIM_OBJ
        self.n_wps = cfg.OBJ_POSE_DATA.NUM_WPS
        self.dim_wp = cfg.OBJ_POSE_DATA.DIM_WP
        use_inter_loss = cfg.TRAIN.USE_INTER_LOSS
        assert cfg.NETWORK.CONFIDENCE_METHOD in ['no', 'concat', 'modulate']

        if is_train:
            print('dim: {}'.format(cfg.NETWORK.TRANSFORM_DIM))
        self.T = cfg.NETWORK.TEMPORAL_LENGTH
        N_before_mhf = cfg.NETWORK.NUM_CHANNELS // cfg.NETWORK.TRANSFORM_DIM
        N_after_mhf = cfg.NETWORK.AFTER_MHF_DIM // cfg.NETWORK.TRANSFORM_DIM
        # pre-embeding hidden_dim from num_joints*2 to hidden/2


        # resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
        #                34: (BasicBlock, [3, 4, 6, 3]),
        #                50: (Bottleneck, [3, 4, 6, 3]),
        #                101: (Bottleneck, [3, 4, 23, 3]),
        #                152: (Bottleneck, [3, 8, 36, 3])}
        # num_layers = cfg.POSE_RESNET.NUM_LAYERS
        # block_class, layers = resnet_spec[num_layers]
        # self.feature_extractor = PoseResNet(block_class, layers, cfg, **kwargs)

        #weights = ResNet18_Weights.DEFAULT
        #self.resnet18 = resnet18(weights=weights, progress=False).eval()
        #self.transforms = weights.transforms()
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(cfg.NETWORK.DROPOUT)
        self.hrnet = HRNet(c=32, nof_joints=self.n_wps)
        """
        self.hr_b_c1 = nn.Conv2d(32, 64, kernel_size=(2, 2), stride=(2, 2))
        self.hr_b_b1 = nn.BatchNorm2d(64, momentum=momentum)
        self.hr_b_c2 = nn.Conv2d(64, 128, kernel_size=(2, 2), stride=(2, 2))
        self.hr_b_b2 = nn.BatchNorm2d(128, momentum=momentum)
        self.hr_b_c3 = nn.Conv2d(128, 128, kernel_size=(2, 2), stride=(2, 2))
        self.hr_b_b3 = nn.BatchNorm2d(128, momentum=momentum)
        self.hr_b_c4 = nn.Conv2d(128, 128, kernel_size=(2, 2), stride=(2, 2))
        self.hr_b_b4 = nn.BatchNorm2d(128, momentum=momentum)
        self.hrnet_bottleneck = nn.Sequential(self.hr_b_c1, self.hr_b_b1, self.relu, self.drop, self.hr_b_c2, self.hr_b_b2, self.relu, self.drop,  self.hr_b_c3, self.hr_b_b3, self.relu, self.drop, self.hr_b_c4, self.hr_b_b4, self.relu, self.drop )
        #self.resnet18 = resnet18()"""

        #self.hrnet_mapping = torch.nn.Linear(4*4*128, self.n_wps*(16+1))
        self.hrnet_mapping=torch.nn.Linear(4096, 1000)
        #self.resnet_mapping=torch.nn.Linear(1000, self.n_wps*(16+1))#+1 for visibility
        self.resnet_mapping = torch.nn.Linear(1000, (16 + 1))
        #self.bn_resnet_mapping=nn.BatchNorm1d(self.n_wps*(16+1), momentum=momentum)
        self.bn_resnet_mapping = nn.BatchNorm1d((16 + 1), momentum=momentum)

        self.head_2D_pose_l1 = torch.nn.Linear(self.n_wps*(16+1), self.n_wps*(16+1))

        self.bn_head_2D_pose_l1 = nn.BatchNorm1d(self.n_wps*(16+1), momentum=momentum)

        self.head_2D_pose_l2 = torch.nn.Linear(self.n_wps * (16 + 1), self.n_wps * 2)
        #print(self.resnet18)
        self.pre_embedding = pre_proj(cfg, in_N=self.n_objs, h_N=N_before_mhf, dropout=dropout, is_train=is_train)
        # proposed mhf

        self.mhf = MHF(cfg, in_channels=self.in_channels, emb_size=self.hidden, T=self.T, dropout=dropout, num_joints=self.n_objs, istrain=is_train)
        if self.cfg.NETWORK.USE_MFT:
            self.fuse_model = FuseView(cfg, N_after_mhf, dropout=cfg.NETWORK.DROPOUT, is_train=is_train, T=self.T,
                                       num_joints=self.n_objs)


        if is_train and use_inter_loss:
            self.shrink_1 = Pose3dShrinkOther(cfg, N=N_after_mhf, channels=cfg.NETWORK.NUM_CHANNELS,
                                              dropout=cfg.NETWORK.DROPOUT, is_train=is_train,
                                              num_joints=self.n_wps)
        self.shrink_out = Pose3dShrink_out(cfg=cfg, N=N_after_mhf, channels=cfg.NETWORK.NUM_CHANNELS,
                                                  dropout=cfg.NETWORK.DROPOUT, is_train=is_train,
                                                  num_joints=self.n_wps)
        self.use_inter_loss = use_inter_loss

        if self.cfg.NETWORK.USE_FEATURE_TRAN and is_train and self.cfg.NETWORK.M_FORMER.MODE == 'mtf':
            if not self.cfg.NETWORK.USE_GT_TRANSFORM:
                if self.cfg.TRAIN.USE_ROT_LOSS and self.training:
                    self.tran_shrink = RotShrink(cfg, N=N_after_mhf, channels=cfg.NETWORK.NUM_CHANNELS,
                                                 dropout=cfg.NETWORK.DROPOUT, is_train=is_train,
                                                 num_joints=self.n_wps)
            else:
                if self.cfg.TRAIN.USE_ROT_LOSS and self.cfg.NETWORK.M_FORMER.GT_TRANSFORM_RES:
                    self.tran_shrink = RotShrink(cfg, N=N_after_mhf, channels=cfg.NETWORK.NUM_CHANNELS,
                                                 dropout=cfg.NETWORK.DROPOUT, is_train=is_train,
                                                 num_joints=self.n_wps)

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
        #B, T, V, C, N = pos_2d.shape
        pos_2d = pos_2d.contiguous()
        pos_2d = torch.squeeze((pos_2d.permute(0, 5, 1, 2, 3, 4).contiguous())).view(-1, 3, 256, 256).contiguous()
        im_feats = self.hrnet(pos_2d)
        #im_feats = self.hrnet_bottleneck(im_feats[0])
        print(im_feats.shape)
        im_feats = im_feats.view(pos_2d.shape[0]*self.n_wps, 4096).contiguous()
        #im_feats = im_feats.view(pos_2d.shape[0], 128*4*4).contiguous()
        print(im_feats.shape)
        im_feats = self.hrnet_mapping(im_feats)


        features_2D = self.drop(self.relu(self.bn_resnet_mapping(self.resnet_mapping(im_feats))))
        print(features_2D.shape)
        features_2D = features_2D.view(pos_2d.shape[0],self.n_wps, (16+1)).contiguous()
        features_2D = features_2D.view(pos_2d.shape[0], self.n_wps*(16 + 1)).contiguous()
        print(features_2D.shape)

        #input()
        keypoints_2D = self.relu(self.bn_head_2D_pose_l1(self.head_2D_pose_l1(features_2D.detach()))) #self.drop(self.relu(self.bn_head_2D_pose_l1(self.head_2D_pose_l1(features_2D))))
        keypoints_2D = self.head_2D_pose_l2(keypoints_2D)
        keypoints_2D = keypoints_2D.view(pos_2d.shape[0], self.n_wps, 2).contiguous()
        keypoints_2D = keypoints_2D.view(-1, 4, self.n_wps, 2).contiguous()
        keypoints_2D = torch.unsqueeze(keypoints_2D, 1)
        keypoints_2D = keypoints_2D.permute(0, 1, 3, 4, 2)
        #keypoints_2D=keypoints_2D.view(-1, 1, self.n_wps, 2, 4).contiguous()
        #features_2D = features_2D.view(-1, 1, self.n_wps, self.dim_obj + 1, 4).contiguous()
        features_2D = features_2D.view(pos_2d.shape[0], self.n_wps, self.dim_obj + 1).contiguous()
        features_2D = features_2D.view(-1, 4, self.n_wps, self.dim_obj + 1).contiguous()
        features_2D = torch.unsqueeze(features_2D, 1)
        features_2D = features_2D.permute(0, 1, 3, 4, 2)
        f = self.pre_embedding(features_2D)
        f = self.mhf(f)  # (B, K, D, T, N)
        #print("f_.shape={}".format(f.shape))
        if self.training and self.use_inter_loss:
            out_1 = self.shrink_1(f[:, :, :self.cfg.NETWORK.TRANSFORM_DIM].contiguous())
        tran = None
        rot = None
        f_fuse_before = f

        if self.cfg.NETWORK.USE_MFT:
            f, tran, att, tran_rot, f_tmp_rcpe, mask = self.fuse_model(f, features_2D, rotation)
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
            return out, keypoints_2D, [out_1, out_1] if self.cfg.NETWORK.USE_MFT else [out_1], tran, rot
        else:
            return out, keypoints_2D, [f_fuse_before]

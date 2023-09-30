from gettext import translation
import numpy as np
import itertools
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys, os
import errno
import copy
import time
import math
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv
from tensorboardX import SummaryWriter

from common.arguments import parse_args
from common.utils import deterministic_random, save_model, save_model_epoch
from common.camera import *
from common.multiview_model import get_models
from common.loss import *
from common.generators import *
from common.data_augmentation_multi_view import *
from common.h36m_dataset import Human36mCamera, Human36mDataset
from common.set_seed import *
from common.config import config as cfg
from common.config import reset_config, update_config
from common.vis import *

set_seed()

args = parse_args()
update_config(args.cfg) ###config file->cfg
reset_config(cfg, args) ###arg -> cfg
print(cfg)
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(cfg.GPU)

print('p2d detector:{}'.format('gt_p2d' if cfg.DATA.USE_GT_2D else cfg.H36M_DATA.P2D_DETECTOR))
HumanCam = Human36mCamera(cfg)


"""keypoints = {}
for sub in [1, 5, 6, 7, 8, 9, 11]:
    if cfg.H36M_DATA.P2D_DETECTOR == 'cpn' or cfg.H36M_DATA.P2D_DETECTOR == 'gt':
        data_pth = 'data/h36m_sub{}.npz'.format(sub)
    elif cfg.H36M_DATA.P2D_DETECTOR == 'ada_fuse':
        data_pth = 'data/h36m_sub{}_ada_fuse.npz'.format(sub)
    
    keypoint = np.load(data_pth, allow_pickle=True)
    lst = keypoint.files
    keypoints_metadata = keypoint['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    keypoints['S{}'.format(sub)] = keypoint['positions_2d'].item()['S{}'.format(sub)]"""

keypoints = dict(np.load(cfg.OBJ_POSE_DATA.ROOT_DIR+'/sim_all_formal.npz', allow_pickle=True))['pose'].item()

print("keypoints.keys() : "+str(keypoints.keys()))

"""for index, (key, value) in enumerate(raw_keypoints.items()):
keypoint = value
#keypoints_metadata = keypoint['metadata'].item()
#keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
keypoints[key] = value.item()"""
'''for index_, (key_, value_) in enumerate(raw_keypoints[key].item()):
    keypoints[key][key_] = value'''

#print(keypoints.keys())
#kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
#joints_left, joints_right = [kps_left, kps_right]
#N_frame_action_dict={}
action_frames = {}
for act in cfg.OBJ_POSE_DATA.SUBJECTS_TRAIN:
    action_frames[act] = 0

for index_tsk, (tsk, exemples) in enumerate(keypoints.items()):
    if tsk in cfg.OBJ_POSE_DATA.SUBJECTS_TRAIN:
        for index_ex, (ex, traj) in enumerate(exemples.items()):
            #N_frame_action_dict[traj[0].shape[0]] = tsk
            action_frames[tsk] += traj[0].shape[0]
#print(N_frame_action_dict)

"""N_frame_action_dict = {
2699:'Directions',2356:'Directions',1552:'Directions',
5873:'Discussion', 5306:'Discussion',2684:'Discussion',2198:'Discussion',
2686:'Eating', 2663:'Eating',2203:'Eating',2275 :'Eating',
1447:'Greeting', 2711:'Greeting',1808:'Greeting', 1695:'Greeting',
3319:'Phoning',3821:'Phoning',3492:'Phoning',3390:'Phoning',
2346:'Photo',1449:'Photo',1990:'Photo',1545:'Photo',
1964:'Posing', 1968:'Posing',1407:'Posing',1481 :'Posing',
1529:'Purchases', 1226:'Purchases',1040:'Purchases', 1026:'Purchases',
2962:'Sitting', 3071:'Sitting',2179:'Sitting', 1857:'Sitting',
2932:'SittingDown', 1554:'SittingDown',1841:'SittingDown', 2004:'SittingDown',
4334:'Smoking',4377:'Smoking',2410:'Smoking',2767:'Smoking',
3312:'Waiting', 1612:'Waiting',2262:'Waiting', 2280:'Waiting',
2237:'WalkDog', 2217:'WalkDog',1435:'WalkDog', 1187:'WalkDog',
1703:'WalkTogether',1685:'WalkTogether',1360:'WalkTogether',1793:'WalkTogether',
1611:'Walking', 2446:'Walking',1621:'Walking', 1637:'Walking',
}"""

"""actions = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Photo', 'Posing', 'Purchases','Sitting','SittingDown', 'Smoking', 'Waiting', 'WalkDog', 'WalkTogether', 'Walking']
train_actions = actions
test_actions = actions
vis_actions = actions"""
actions = []
train_actions = []
test_actions = []
vis_actions = []
for k in range(300):
    ex = "exemple_"+str(k)
    actions.append(ex)
    if k<int(300*0.8):
        train_actions.append(ex)
    else:
        test_actions.append(ex)
        vis_actions.append(ex)



"""for k,v in N_frame_action_dict.items():
    action_frames[v] += k"""
if cfg.H36M_DATA.P2D_DETECTOR == 'cpn' or cfg.H36M_DATA.P2D_DETECTOR == 'gt':
    vis_score = pickle.load(open('./data/score.pkl', 'rb'))
elif cfg.H36M_DATA.P2D_DETECTOR[:3] == 'ada':
    vis_score = pickle.load(open('./data/vis_ada.pkl', 'rb'))

def fetch(subjects, action_filter=None,  parse_3d_poses=True, is_test = False):
    out_poses_3d = []
    out_poses_2d_view1 = []
    out_poses_2d_view2 = []
    out_poses_2d_view3 = []
    out_poses_2d_view4 = []
    out_camera_params = []
    out_subject_action = []
    used_cameras = cfg.H36M_DATA.TEST_CAMERAS if is_test else cfg.H36M_DATA.TRAIN_CAMERAS
    #print(keypoints.keys())
    for subject in subjects:
        """print(subject)
        print("keypoints[subject]['exemple_0'][0] : " + str(keypoints[subject]['exemple_0'][0][0]))

        input()"""
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue
                
            poses_2d = keypoints[subject][action]

            out_subject_action.append([subject, action])
            n_frames = poses_2d[0].shape[0]
            if cfg.DATA.HAS_2D_PREDS:
                vis_name_1 = '{}_{}.{}'.format(subject, action, 0)
                vis_name_2 = '{}_{}.{}'.format(subject, action, 1)
                vis_name_3 = '{}_{}.{}'.format(subject, action, 2)
                vis_name_4 = '{}_{}.{}'.format(subject, action, 3)
                vis_score_cam0 = vis_score[vis_name_1][:n_frames][...,np.newaxis]
                vis_score_cam1 = vis_score[vis_name_2][:n_frames][...,np.newaxis]
                vis_score_cam2 = vis_score[vis_name_3][:n_frames][...,np.newaxis]
                vis_score_cam3 = vis_score[vis_name_4][:n_frames][...,np.newaxis]
                if vis_score_cam3.shape[0] != vis_score_cam2.shape[0]:
                    vis_score_cam2 = vis_score_cam2[:-1]
                    vis_score_cam1 = vis_score_cam1[:-1]
                    vis_score_cam0 = vis_score_cam0[:-1]
                    for i in range(4):
                        poses_2d[i] = poses_2d[i][:-1]
                out_poses_2d_view1.append(np.concatenate((poses_2d[0], vis_score_cam0), axis =-1))
                out_poses_2d_view2.append(np.concatenate((poses_2d[1], vis_score_cam1), axis =-1))
                out_poses_2d_view3.append(np.concatenate((poses_2d[2], vis_score_cam2), axis =-1))
                out_poses_2d_view4.append(np.concatenate((poses_2d[3], vis_score_cam3), axis =-1))
            else:
                out_poses_2d_view1.append(poses_2d[0])
                out_poses_2d_view2.append(poses_2d[1])
                out_poses_2d_view3.append(poses_2d[2])
                out_poses_2d_view4.append(poses_2d[3])

    
    final_pose = []
    if 0 in used_cameras:
        final_pose.append(out_poses_2d_view1)
    if 1 in used_cameras:
        final_pose.append(out_poses_2d_view2)
    if 2 in used_cameras:
        final_pose.append(out_poses_2d_view3)
    if 3 in used_cameras:
        final_pose.append(out_poses_2d_view4)
        
    if is_test is True:
        return final_pose
    else:
        return final_pose, out_subject_action

use_2d_gt = cfg.DATA.USE_GT_2D
receptive_field = cfg.NETWORK.TEMPORAL_LENGTH
pad = receptive_field // 2
causal_shift = 0
model, model_test = get_models(cfg)

#####模型参数量、计算量(MACs)、inference time
if cfg.VIS.VIS_COMPLEXITY:
    from thop import profile
    from thop import clever_format
    if args.eval:
        from ptflops import get_model_complexity_info
    #####模型参数量、计算量(MACs)
    receptive_field = 1
    model_test.eval()
    for i in range(1,5):
        input = torch.randn(1, receptive_field,17,3,i)
        rotation = torch.randn(1, 3, 3,receptive_field,i,i)
        macs, params = profile(model_test, inputs=(input, rotation))
        macs, params = clever_format([macs, params], "%.3f")
        print('view: {} T: {} MACs:{} params:{}'.format(i, receptive_field, macs, params))
        if args.eval:
            flops, params = get_model_complexity_info(model_test, (receptive_field,17,3,i), as_strings=True, print_per_layer_stat=False)
            print('Flops:{}, Params:{}'.format(flops, params))
    #####inference time
    infer_model = model_test.cuda()
    infer_model.eval()
    for receptive_field in [1, 27]:
        for i in range(1,5):
            input = torch.randn(1, receptive_field,17,3,i).float().cuda()
            rotation = torch.randn(1, 3, 3,receptive_field,i,i).float().cuda()
            
            for k in range(100):
                out = infer_model(input, rotation)
            
            N = 1000
            torch.cuda.synchronize()
            start_time = time.time()
            for n in range(N):
                infer_model(input, rotation)
            torch.cuda.synchronize()
            end_time = time.time()
            print('n_frames:{} n_views: {}  time:{:.4f}'.format(receptive_field, i, (end_time - start_time) / N))
    exit()
else:
    total_params = sum(p.numel() for p in model_test.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model_test.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

EVAL = args.eval
ax_views = []

if EVAL and cfg.VIS.VIS_3D:
    plt.ion()
    vis_tool = Vis(cfg, 2)
        
def load_state(model_train, model_test):
    train_state = model_train.state_dict()
    test_state = model_test.state_dict()
    pretrained_dict = {k:v for k, v in train_state.items() if k in test_state}
    test_state.update(pretrained_dict)
    model_test.load_state_dict(test_state)
    
if EVAL and not cfg.TEST.TRIANGULATE:
    chk_filename = cfg.TEST.CHECKPOINT
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    model_checkpoint = checkpoint['model'] if 'best_model' not in checkpoint.keys() else checkpoint['best_model']
    train_checkpoint = model.state_dict()
    test_checkpoint = model_test.state_dict()
    for k, v in train_checkpoint.items():
        if k not in model_checkpoint.keys():
            continue
        checkpoint_v = model_checkpoint[k]
        if 'p_shrink.shrink' in k:
            if model_checkpoint[k].shape[0] == 32:
                checkpoint_v = checkpoint_v[1::2]

        train_checkpoint[k] = checkpoint_v

    print('EVAL: This model was trained for {} epochs'.format(checkpoint['epoch']))
    model.load_state_dict(train_checkpoint)

if True:
    if not cfg.DEBUG and (not args.eval or args.log):
        summary_writer = SummaryWriter(log_dir=cfg.LOG_DIR)
    else:
        summary_writer = None
    
    poses_train_2d, subject_action = fetch(cfg.OBJ_POSE_DATA.SUBJECTS_TRAIN, train_actions)

    lr = cfg.TRAIN.LEARNING_RATE
    if cfg.TRAIN.RESUME:
        chk_filename = cfg.TRAIN.RESUME_CHECKPOINT
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        print('RESUME: This model was trained for {} epochs'.format(checkpoint['epoch']))
        model.load_state_dict(checkpoint['model'])
    if torch.cuda.is_available() and not cfg.TEST.TRIANGULATE:
        model = torch.nn.DataParallel(model).cuda()
        model_test = torch.nn.DataParallel(model_test).cuda()
    optimizer = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=lr, amsgrad=True)    
    if cfg.TRAIN.RESUME:
        epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_result_epoch = checkpoint['best_epoch']
        best_state_dict = checkpoint['best_model']
        lr = checkpoint['lr']
        best_result = 100
    else:
        epoch = 0
        best_result = 100
        best_state_dict = None
        best_result_epoch = 0
        
    lr_decay = cfg.TRAIN.LR_DECAY
    initial_momentum = 0.1
    final_momentum = 0.001
    train_generator = ChunkedGenerator(cfg.TRAIN.BATCH_SIZE, poses_train_2d, 1,pad=pad, causal_shift=causal_shift, shuffle=True, augment=False, sub_act=subject_action) if cfg.H36M_DATA.PROJ_Frm_3DCAM == True else ChunkedGenerator(cfg.TRAIN.BATCH_SIZE, poses_train_2d, 1,pad=pad, causal_shift=causal_shift, shuffle=True, augment=False)
 
    print('** Starting.')
    
    data_aug = DataAug(cfg, add_view = cfg.TRAIN.NUM_AUGMENT_VIEWS)
    iters = 0
    msefun = torch.nn.L1Loss() 
    num_train_views = len(cfg.H36M_DATA.TRAIN_CAMERAS) + cfg.TRAIN.NUM_AUGMENT_VIEWS
    """np.save("mean_3D_cam_720_ex.npy")
    np.save("std_3D_cam_720_ex.npy")"""
    min_3D_cam_720_ex = torch.from_numpy(np.load("min_3D_cam_720_ex.npy")).cuda()
    max_3D_cam_720_ex = torch.from_numpy(np.load("max_3D_cam_720_ex.npy")).cuda()
    range_3D_cam_720_ex = max_3D_cam_720_ex-min_3D_cam_720_ex

    min_gt_obj_center_720_ex = torch.from_numpy(np.load("min_gt_obj_center_720_ex.npy")).cuda()
    max_gt_obj_center_720_ex = torch.from_numpy(np.load("max_gt_obj_center_720_ex.npy")).cuda()
    range_gt_obj_center_720_ex = max_gt_obj_center_720_ex - min_gt_obj_center_720_ex

    min_gt_shifted_obj_720_ex = torch.from_numpy(np.load("min_gt_shifted_obj_720_ex.npy")).cuda()
    max_gt_shifted_obj_720_ex = torch.from_numpy(np.load("max_gt_shifted_obj_720_ex.npy")).cuda()
    range_gt_shifted_obj_720_ex = max_gt_shifted_obj_720_ex - min_gt_shifted_obj_720_ex



    extrinsics = []
    intrinsics = []
    full_transf = []
    """ same cam order as in generation code : """
    for cam in ["over_shoulder_left", "over_shoulder_right", "overhead", "front"]:
        cam_ext_path = cfg.OBJ_POSE_DATA.ROOT_DIR+"/_cam_"+cam+"_extrinsics.npy"
        extrinsics_mtx = np.load(cam_ext_path)
        C = np.expand_dims(extrinsics_mtx[:3, 3], 0).T
        R = extrinsics_mtx[:3, :3]
        R_inv = R.T  # inverse of rot matrix is transpose
        R_inv_C = np.matmul(R_inv, C)
        extrinsics_mtx = np.concatenate((R_inv, -R_inv_C), -1)

        cam_int_path = cfg.OBJ_POSE_DATA.ROOT_DIR + "/_cam_" + cam + "_intrinsics.npy"
        intrinsics_mtx = np.load(cam_int_path)

        transf = np.concatenate([np.matmul(intrinsics_mtx, extrinsics_mtx), np.array([[0,0,0,1]])], axis=0)

        full_transf.append(np.expand_dims(transf, 0))

        extrinsics_mtx = np.concatenate([extrinsics_mtx, np.array([[0, 0, 0, 1]])], axis=0)
        intrinsics_mtx = np.concatenate([intrinsics_mtx, np.array([[0], [0], [0]])], axis=-1)
        extrinsics.append(np.expand_dims(extrinsics_mtx, 0))
        intrinsics.append(np.expand_dims(intrinsics_mtx, 0))

    extrinsics = np.concatenate(extrinsics, axis=0)
    extrinsics = torch.from_numpy(extrinsics.astype('float32')).cuda()
    intrinsics = np.concatenate(intrinsics, axis=0)
    intrinsics = torch.from_numpy(intrinsics.astype('float32')).cuda()
    full_transf = np.concatenate(full_transf, axis=0)
    full_transf =  torch.from_numpy(full_transf.astype('float32')).cuda()


    x_idxs = np.expand_dims([0, 0, 0, 0, 1, 1, 1, 1], -1)
    y_idxs =  np.expand_dims([0,0,1,1,0,0,1,1], -1)+2
    z_idxs = np.expand_dims([0,1,0,1,0,1,0,1], -1)+4
    point_indexes = np.concatenate([x_idxs, y_idxs, z_idxs], axis=-1).tolist()


    while epoch < cfg.TRAIN.NUM_EPOCHES:
        start_time = time.time()
        model.train()
        process = tqdm(total = train_generator.num_batches)
        #deb_idx=0
        for batch_2d, sub_action in train_generator.next_epoch():
            """if deb_idx>2:
                break
            deb_idx+=1"""

            if EVAL:
                break
            process.update(1)
            inputs = torch.from_numpy(batch_2d.astype('float32'))
            """print(inputs.shape)
            print(inputs[0, 0, 0, :, 0])
            input()"""
            #assert inputs.shape[-2] == 8 #(p2d_gt, p2d_pre, p3d, vis)
            if cfg.DATA.HAS_2D_PREDS:
                inputs_2d_gt = inputs[..., :, :cfg.NETWORK.INPUT_DIM, :]/256
                inputs_2d_pre = inputs[..., cfg.NETWORK.INPUT_DIM:2*cfg.NETWORK.INPUT_DIM, :]/256
                dims_sum = 2*cfg.NETWORK.INPUT_DIM+cfg.OBJ_POSE_DATA.DIM_JOINT
                cam_3d = inputs[..., 2*cfg.NETWORK.INPUT_DIM:dims_sum, :]
            else:
                inputs_2d_gt = inputs[..., :, :cfg.NETWORK.INPUT_DIM, :]#/256
                """print(torch.mean(inputs_2d_gt))
                print(torch.std(inputs_2d_gt))
                print(torch.min(inputs_2d_gt))
                print(torch.max(inputs_2d_gt))
                input()"""
                dims_sum = cfg.NETWORK.INPUT_DIM + cfg.OBJ_POSE_DATA.DIM_JOINT
                cam_3d = inputs[..., cfg.NETWORK.INPUT_DIM:dims_sum, :]
                inputs_2d_gt[..., 1:cfg.NETWORK.INPUT_DIM, :] = inputs_2d_gt[..., 1:cfg.NETWORK.INPUT_DIM, :] / 256
                inputs_2d_gt[..., 0, :] = inputs_2d_gt[..., 0, :] / 15

            B, T, V, C, N = cam_3d.shape
            if use_2d_gt or (not cfg.DATA.HAS_2D_PREDS) :
                vis = torch.ones(B, T, V, 1, N)
            else:
                vis = inputs[...,-1, :]
            if cfg.TRAIN.NUM_AUGMENT_VIEWS:
                vis = torch.cat((vis, torch.ones(B, T, V, 1, cfg.TRAIN.NUM_AUGMENT_VIEWS)), dim = -1)

            
            inputs_3d_gt = cam_3d.cuda()
            view_list = list(range(num_train_views))
    
            if cfg.TRAIN.NUM_AUGMENT_VIEWS > 0:
                pos_gt_3d_tmp = copy.deepcopy(inputs_3d_gt)
                pos_gt_2d, pos_gt_3d = data_aug(pos_gt_2d = inputs_2d_gt, pos_gt_3d = pos_gt_3d_tmp)
                if cfg.DATA.HAS_2D_PREDS:
                    pos_pre_2d = torch.cat((inputs_2d_pre, pos_gt_2d[...,inputs_2d_pre.shape[-1]:]), dim = -1)
                if use_2d_gt or (not cfg.DATA.HAS_2D_PREDS):
                    h36_inp = pos_gt_2d[..., view_list]
                else:
                    h36_inp = pos_pre_2d[..., view_list]
                pos_gt = pos_gt_3d[..., view_list]

            else:
                if use_2d_gt or (not cfg.DATA.HAS_2D_PREDS):
                    h36_inp = inputs_2d_gt[..., view_list]
                else:
                    h36_inp = inputs_2d_pre[..., view_list]
                pos_gt = inputs_3d_gt[..., view_list]
            #if cfg.H36M_DATA.PROJ_Frm_3DCAM == True:
            #    prj_3dgt_to_2d= HumanCam.p3d_im2d(pos_gt, sub_action, view_list)
            p3d_gt_ori = copy.deepcopy(pos_gt)
            p3d_root = copy.deepcopy(pos_gt[:,:,:1]) #(B,T, 1, 3, N)
            #pos_gt[:,:,:1] = 0
            
            optimizer.zero_grad()
            inp = torch.cat((h36_inp, vis), dim = -2)
            if cfg.NETWORK.USE_GT_TRANSFORM or cfg.TRAIN.USE_ROT_LOSS:
                #相机之间的旋转
                rotation = get_rotation(pos_gt[:,:1]) #(B, 3, 3, 1, N, N)

                # #相机之间的位移
                # #print(rotation)
                # t = torch.einsum('btjcn,bqcmn->btjqmn', p3d_root[:,:1], rotation[:,:,:,0])#(B, T, 1, 3, N, N)
                # t = t - t[...,:1]
                # t = t.permute(0, 2, 3, 1, 4, 5) #(B, 1, 3, T, N, N)
                # if cfg.NETWORK.M_FORMER.GT_TRANSFORM_MODE == 'rt':
                #     rotation = torch.cat((rotation, t), dim = 1)
            else:
                rotation = None
 
            if cfg.TRAIN.USE_INTER_LOSS:
                print('Input shape is {}'.format(inp.shape))
                out, other_out, tran, pred_rot = model(inp, rotation) #mask:(B, 1, 1, 1, N, N)
            else:
                out = model(inp, rotation)
            
            if cfg.H36M_DATA.PROJ_Frm_3DCAM == True:
                p3d_gt_abs = pos_gt+p3d_root
                prj_3dpre_to_2d = HumanCam.p3d_im2d_batch(out, sub_action, view_list, with_distor=True)
                prj_3dgt_rela_to_2d = HumanCam.p3d_im2d_batch(p3d_gt_ori[:, pad:pad+1], sub_action, view_list, with_distor=True)
                prj_3dgt_abs_to_2d = HumanCam.p3d_im2d_batch(p3d_gt_abs[:, pad:pad+1], sub_action, view_list, with_distor=False)
                prj_2dgt_to_3d = HumanCam.p2d_cam3d_batch(inputs_2d_gt[:, pad:pad+1, :, :], sub_action, view_list[:4])
                for vw in range(4):
                    print("------------- View {} ----------------\n".format(str(vw)))
                    print('Network Output is {}; \n Projection Output is {}'.format(out[0,0,:,:,vw], prj_3dpre_to_2d[0,vw,0]))
                    print("GT 2D : "+str(inp[0,3:4,:,:,vw]))

            out = out.permute(0, 1, 4, 2,3).contiguous() #(B, T, N, J. C)
            pos_gt = pos_gt.permute(0, 1, 4,2, 3).contiguous()
            """if cfg.TRAIN.USE_INTER_LOSS:
                for i in range(len(other_out)): 
                    other_out[i] = other_out[i].permute(0, 1, 4, 2,3).contiguous() #(B, T, N, J. C)
            """
            """print("extrinsics.shape : "+str(extrinsics.shape))
            print("pos_gt[:,pad:pad+1].shape : "+str(pos_gt[:,pad:pad+1].shape))
            print(point_indexes)"""
            """print("pos_gt[:,:][0,:,0,0] : "+str(pos_gt[:,:][0,:,0,0]))
            pt_3D_0000 = (pos_gt[:,pad:pad+1])[0,0,0,0].cpu().detach().numpy()
            print(pt_3D_0000.shape)
            for x_i in [0, 1]:
                for y_i in [2, 3]:
                    for z_i in [4, 5]:
                        _pose = np.concatenate([pt_3D_0000[[x_i, y_i, z_i]], [1]])
                        pt = np.matmul(full_transf[0].cpu().detach().numpy(), _pose)
                        pt[:2] = pt[:2]/pt[2]
                        print(pt)"""

            points_3d_world = pos_gt[:,pad:pad+1][:, :, :, :, point_indexes] # (B, T, N, J. N_pts C)



            init_shape = points_3d_world.shape
            """print(points_3d_world.shape) #(..., views, n_objects, object_dim)
            print(inputs_2d_gt[0, 0, :, 0, 0]*256)

            x = points_3d_world[0, 0, 0, :, :, 0].detach().cpu().numpy()
            y = points_3d_world[0, 0, 0, :, :, 1].detach().cpu().numpy()
            z = points_3d_world[0, 0, 0, :, :, 2].detach().cpu().numpy()
            fig = plt.figure("world space")
            ax = plt.axes(projection='3d')
            ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
            ax.scatter3D(x, y, z)"""

            points_3d_world = torch.cat((points_3d_world, torch.ones(points_3d_world.shape[:-1]+(1,)).cuda()), axis=-1)
            points_3d_world = torch.permute(points_3d_world.view(*points_3d_world.shape[:3], -1, 4), (0,1,3,2,4)).contiguous()

            # data gen projection on object 0, view 0


            #print(points_3d_world.shape)
            """print(points_3d_world.shape)
            print(full_transf.shape)
            input()
            point_im_cam = torch.einsum('voi, btpvi->btpvo', full_transf, points_3d_world)
            point_im_cam = torch.permute(point_im_cam, (0, 1, 3, 2, 4)).contiguous()
            point_im_cam = point_im_cam.view(*init_shape[:-1], 4)
            print(point_im_cam.shape)
            point_im_cam = torch.divide(point_im_cam[:, :, :, :, :, :2], point_im_cam[:, :, :, :, :, 2:3])
            print("point_im_cam[0, 0, 0, 0] : " + str(point_im_cam[0, 0, 0, 0]))
            input()"""
            point_3D_cam = torch.einsum('voi, btpvi->btpvo', extrinsics , points_3d_world)
            """print(point_3D_cam.shape)
            print(point_3D_cam[0, 0, 0, 0])
            print("intrinsics.shape : " + str(intrinsics.shape))"""
            """point_im_cam = torch.einsum('voi, btpvi->btpvo', intrinsics, point_3D_cam)
            point_im_cam = torch.permute(point_im_cam, (0, 1, 3, 2, 4)).contiguous()
            print("init shape : "+str(init_shape))
            print("point_3D_cam shape : "+str(point_im_cam.shape))
            point_im_cam = point_im_cam.view(*init_shape)
            point_im_cam = torch.divide(point_im_cam[:, :, :, :, :, :2], point_im_cam[:, :, :, :, :, -1:])
            print("point_im_cam[0, 0, 0, 0] : " + str(point_im_cam.shape))
            print("point_im_cam[0, 0, 0, 0] : "+str(point_im_cam[0, 0, 0, 0]))
            print("2g gt : " + str(inputs_2d_gt.shape))
            print("2g gt : " + str(inputs_2d_gt[0, 0, :, :, 0]))
            input()"""
            point_3D_cam = torch.permute(point_3D_cam, (0,1,3,2,4)).contiguous()

            point_3D_cam = point_3D_cam.view(*init_shape[:-1], 4)
            """print(point_3D_cam.shape)
            x = point_3D_cam[0, 0, 0, :, :, 0].detach().cpu().numpy()
            y = point_3D_cam[0, 0, 0, :, :, 1].detach().cpu().numpy()
            z = point_3D_cam[0, 0, 0, :, :, 2].detach().cpu().numpy()
            fig = plt.figure("cam space")
            ax = plt.axes(projection='3d')
            ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
            ax.scatter3D(x, y, z)
            plt.show()"""
            #print("starting transforms")
            point_3D_cam_unnormed = point_3D_cam[:, :, :, :, :, :3]
            x_gt_obj_center = torch.mean(point_3D_cam[:, :, :, :, :, 0], dim=-1, keepdim=True)
            y_gt_obj_center = torch.mean(point_3D_cam[:, :, :, :, :, 1], dim=-1, keepdim=True)
            z_gt_obj_center = torch.mean(point_3D_cam[:, :, :, :, :, 2], dim=-1, keepdim=True)
            point_3D_cam = point_3D_cam[:, :, :, :, :, :3]
            #print(1)
            gt_obj_center = torch.cat((x_gt_obj_center, y_gt_obj_center, z_gt_obj_center), -1)
            gt_obj_center=  torch.unsqueeze(gt_obj_center, -2)
            """print(point_3D_cam.shape)
            print(gt_obj_center.shape)"""
            gt_shifted_obj = point_3D_cam - gt_obj_center

            """print(range_gt_obj_center_720_ex)
            input()"""
            #point_3D_cam = (point_3D_cam[:, :, :, :, :, :3]-min_3D_cam_720_ex)/range_3D_cam_720_ex
            gt_obj_center = torch.where(range_gt_obj_center_720_ex!=0, (gt_obj_center-min_gt_obj_center_720_ex)/range_gt_obj_center_720_ex, 0)
            #gt_obj_center = (gt_obj_center-min_gt_obj_center_720_ex)/range_gt_obj_center_720_ex
            gt_shifted_obj = torch.where(range_gt_shifted_obj_720_ex!=0, (gt_shifted_obj-min_gt_shifted_obj_720_ex)/range_gt_shifted_obj_720_ex, 0)
            #gt_shifted_obj = (gt_shifted_obj-min_gt_shifted_obj_720_ex)/range_gt_shifted_obj_720_ex

            """x = gt_shifted_obj[0, 0, 0, :, :, 0].detach().cpu().numpy()
            y = gt_shifted_obj[0, 0, 0, :, :, 1].detach().cpu().numpy()
            z = gt_shifted_obj[0, 0, 0, :, :, 2].detach().cpu().numpy()
            print((x, y, z))
            fig = plt.figure("shifted obj")
            ax = plt.axes(projection='3d')
            ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
            ax.scatter3D(x, y, z)

            x = gt_obj_center[0, 0, 0, :, :, 0].detach().cpu().numpy()
            y = gt_obj_center[0, 0, 0, :, :, 1].detach().cpu().numpy()
            z = gt_obj_center[0, 0, 0, :, :, 2].detach().cpu().numpy()
            fig = plt.figure("centers")
            ax = plt.axes(projection='3d')
            ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
            ax.scatter3D(x, y, z)

            plt.show()"""

            """np.save("mean_gt_shifted_obj_720_ex.npy",(np.mean(gt_shifted_obj[:, :, :, :, :, :3].detach().cpu().numpy(), axis=(0, 1, 4), keepdims=True)))
            np.save("std_gt_shifted_obj_720_ex.npy",np.std(gt_shifted_obj[:, :, :, :, :, :3].detach().cpu().numpy(), axis=(0, 1, 4), keepdims=True))
            np.save("min_gt_shifted_obj_720_ex.npy",np.min(gt_shifted_obj[:, :, :, :, :, :3].detach().cpu().numpy(), axis=(0, 1, 4), keepdims=True))
            np.save("max_gt_shifted_obj_720_ex.npy",np.max(gt_shifted_obj[:, :, :, :, :, :3].detach().cpu().numpy(), axis=(0, 1, 4), keepdims=True))
            
            np.save("mean_gt_obj_center_720_ex.npy",(np.mean(gt_obj_center[:, :, :, :, :, :3].detach().cpu().numpy(), axis=(0, 1, 3, 4), keepdims=True)))
            np.save("std_gt_obj_center_720_ex.npy",np.std(gt_obj_center[:, :, :, :, :, :3].detach().cpu().numpy(), axis=(0, 1, 3,  4), keepdims=True))
            np.save("min_gt_obj_center_720_ex.npy",np.min(gt_obj_center[:, :, :, :, :, :3].detach().cpu().numpy(), axis=(0, 1, 3, 4), keepdims=True))
            np.save("max_gt_obj_center_720_ex.npy",np.max(gt_obj_center[:, :, :, :, :, :3].detach().cpu().numpy(), axis=(0, 1, 3,  4), keepdims=True))
            """
            #input()

            #print(point_3D_cam.shape)
            #print(point_3D_cam[0, 0, 0, 0])
            """print(out[:, :, :, :, point_indexes].shape)
            print(point_3D_cam[:, :, :, :, :, :3].shape)
            input()"""
            #loss = mpjpe(out , pos_gt[:,pad:pad+1])
            loss_shifted_box= mpjpe(out[:, :, :, :, point_indexes], gt_shifted_obj)
            loss_center = mpjpe(out[:, :, :, :, [6, 7, 8]], gt_obj_center[:, :, :, :, 0, :])

            #print(torch.unsqueeze(out[:, :, :, :, point_indexes], 4).shape)
            pts_a = torch.unsqueeze(out[:, :, :, :, point_indexes], 4).repeat((1,1,1,1,8,1,1))
            #print(pts_a.shape)
            pts_b = torch.permute(pts_a, (0,1,2,3, 5, 4, 6)).contiguous()

            #print(pts_b.shape)
            distances_out = torch.norm(pts_a - pts_b, dim=-1)
            #print(distances_out.shape)
            pts_a = torch.unsqueeze(gt_shifted_obj, 4).repeat((1,1,1,1,8,1,1))
            pts_b = torch.permute(pts_a, (0,1,2,3, 5, 4, 6)).contiguous()
            distances_gt = torch.norm(pts_a - pts_b, dim=-1)
            #print(distances_gt.shape)
            loss_pts_distances = mpjpe(distances_out, distances_gt)
            #input()
            #loss = mpjpe(out[:, :, :, :, point_indexes], gt_obj_center)
            loss = loss_shifted_box + 2*loss_center + loss_pts_distances
            pred_center = torch.unsqueeze(out[:, :, :, :, [6, 7, 8]], -2)*range_gt_obj_center_720_ex+min_gt_obj_center_720_ex
            pred_shift_center = out[:, :, :, :, point_indexes]*range_gt_shifted_obj_720_ex+min_gt_shifted_obj_720_ex
            un_normed_pred = pred_shift_center+pred_center
            """x = un_normed_pred[0, 0, 0, :, :, 0].detach().cpu().numpy()
            y = un_normed_pred[0, 0, 0, :, :, 1].detach().cpu().numpy()
            z = un_normed_pred[0, 0, 0, :, :, 2].detach().cpu().numpy()
            fig = plt.figure("un_normed_pred")
            ax = plt.axes(projection='3d')
            ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
            ax.scatter3D(x, y, z)

            plt.show()"""
            loss_unnormed = mpjpe(un_normed_pred, point_3D_cam_unnormed)
            if summary_writer is not None:
                summary_writer.add_scalar("loss_final/iter", loss, iters)
                summary_writer.add_scalar("loss_unnormed_final/iter", loss_unnormed, iters)
            if pred_rot is not None and cfg.TRAIN.USE_ROT_LOSS:
                tran_loss = msefun(pred_rot, rotation)

                if summary_writer is not None:
                    summary_writer.add_scalar("loss_tran/iter", tran_loss, iters)
                loss = loss + cfg.TRAIN.ROT_LOSS_WEIGHT * tran_loss
            
            loss_consis_weight = cfg.TRAIN.CONSIS_LOSS_WEIGHT
            loss_consis = 0
            print('Supervised Loss is {}'.format(loss))
            if cfg.H36M_DATA.PROJ_Frm_3DCAM == True:
                prj_3dpre_to_2d = prj_3dpre_to_2d.permute(0,2,3,4,1).contiguous()
                #loss_consis = msefun(prj_3dpre_to_2d, prj_3dgt_to_2d[:,pad:pad+1])
                loss_consis = msefun(prj_3dpre_to_2d, pos_gt_2d[..., [0,1,2,3]][:,pad:pad+1].to(prj_3dpre_to_2d.device))
                print('Consistancy Loss is {}'.format(loss_consis))
                print('Supervised Loss is {}'.format(loss))
                loss = loss #+ loss_consis_weight * loss_consis
                print('Summed Loss is {}'.format(loss))
                summary_writer.add_scalar("loss_consis/iter", loss_consis, iters)
            
            inter_loss_weight = cfg.TRAIN.INTER_LOSS_WEIGHT
            inter_loss_all = 0
            """if cfg.TRAIN.USE_INTER_LOSS:
                for i in range(len(other_out)):
                    if other_out[i].shape[1] == 1:
                        inter_loss = mpjpe(other_out[i] , pos_gt[:,pad:pad+1])
                    else:

                        inter_loss = mpjpe(other_out[i] , pos_gt)
                    inter_loss_all = inter_loss_all + inter_loss_weight[i] * inter_loss 
                    if summary_writer is not None:
                        summary_writer.add_scalar("loss_inter_{}/iter".format(cfg.TRAIN.INTER_LOSS_NAME[i]), inter_loss, iters)
            """
            mv_loss_all = 0
            if cfg.TRAIN.USE_MV_LOSS and epoch >= 0:
                mv_loss = mv_mpjpe(other_out[-1], pos_gt[:,pad:pad+1], mask) if other_out[-1] is not None else mv_mpjpe(other_out[0], pos_gt[:,pad:pad+1], mask)
                mv_loss_all = mv_loss_all + cfg.TRAIN.MV_LOSS_WEIGHT * mv_loss
                if summary_writer is not None:
                    summary_writer.add_scalar("loss_mv_loss/iter", mv_loss, iters)

            loss_total = loss
            """if cfg.TRAIN.USE_INTER_LOSS:
                loss_total = loss_total + inter_loss_all """
            if cfg.TRAIN.USE_MV_LOSS and epoch >= 0:
                loss_total = loss_total + mv_loss_all 
                
            loss_total.backward()

            optimizer.step()
            iters += 1

        process.close() 

        ###########eval
        with torch.no_grad():
            if not cfg.TEST.TRIANGULATE:
                load_state(model, model_test)
                model_test.eval()
            NUM_VIEW = len(cfg.H36M_DATA.TEST_CAMERAS)
            if EVAL:
                eval_start_time = time.time()
            eval_results = np.zeros((len(cfg.TEST.NUM_FRAMES), len(cfg.OBJ_POSE_DATA.SUBJECTS_TRAIN), len(cfg.TEST.NUM_VIEWS), NUM_VIEW, 4))
            t_len_id = 0
            for t_len in cfg.TEST.NUM_FRAMES:
                epoch_loss_valid = 0  
                action_mpjpe = {}
                for subj in cfg.OBJ_POSE_DATA.SUBJECTS_TRAIN:
                    action_mpjpe[subj] = [0] * NUM_VIEW
                    for i in range(NUM_VIEW):
                        action_mpjpe[subj][i] = [0] * (NUM_VIEW + 1)
                N = [0] * NUM_VIEW
                for i in range(NUM_VIEW):
                    N[i] = [0] * (NUM_VIEW + 1)
                num_view_id = 0
                for num_view in cfg.TEST.NUM_VIEWS:
                    pad_t = t_len // 2
                    view_list_id=0
                    for view_list in itertools.combinations(list(range(NUM_VIEW)), num_view):
                        view_list = list(view_list)
                        N[num_view - 1][-1] += 1
                        for i in view_list:
                            N[num_view - 1][i] += 1
                        valid_subject_id=0
                        for valid_subject in cfg.OBJ_POSE_DATA.SUBJECTS_TEST:
                            n_batch = 0
                            for act in vis_actions if EVAL else test_actions:
                                """if n_batch >= 5:
                                    break"""
                                poses_valid_2d = fetch([valid_subject], [act], is_test =True)
                                #print("poses_valid_2d : "+str(len(poses_valid_2d)))
                                test_generator = ChunkedGenerator(cfg.TEST.BATCH_SIZE, poses_valid_2d, 1,pad=pad_t, causal_shift=causal_shift, shuffle=True, augment=False)
                                #batch_eval_results = np.zeros((len(view_list), ))
                                for batch_2d, _ in test_generator.next_epoch():
                                    inputs = torch.from_numpy(batch_2d.astype('float32')).cuda()
                                    if cfg.DATA.HAS_2D_PREDS:
                                        inputs_2d_gt = inputs[..., :cfg.NETWORK.INPUT_DIM, :]/256
                                        inputs_2d_pre = inputs[...,cfg.NETWORK.INPUT_DIM:2*cfg.NETWORK.INPUT_DIM,:]/256
                                        sum_dims = 2*cfg.NETWORK.INPUT_DIM+cfg.OBJ_POSE_DATA.DIM_JOINT
                                        cam_3d = inputs[..., 2*cfg.NETWORK.INPUT_DIM:sum_dims, :]
                                    else:
                                        inputs_2d_gt = inputs[..., :cfg.NETWORK.INPUT_DIM, :]#/256
                                        sum_dims = cfg.NETWORK.INPUT_DIM + cfg.OBJ_POSE_DATA.DIM_JOINT
                                        cam_3d = inputs[..., cfg.NETWORK.INPUT_DIM:sum_dims, :]
                                        inputs_2d_gt[..., 1:cfg.NETWORK.INPUT_DIM, :] = inputs_2d_gt[..., 1:cfg.NETWORK.INPUT_DIM, :]/256
                                        inputs_2d_gt[..., 0, :] = inputs_2d_gt[...,0,:] / 15
                                    vis = inputs[...,-1,:].cuda()
                                    inputs_3d_gt = cam_3d[:,pad_t:pad_t+1]

                                    #inputs_3d_gt[:,:,0] = 0
                                    if use_2d_gt or (not cfg.DATA.HAS_2D_PREDS):
                                        inp = inputs_2d_gt
                                        vis = torch.unsqueeze(torch.ones(*vis.shape), -2).cuda()
                                    else:
                                        inp = inputs_2d_pre

                                    inp = inp[...,view_list] #B, T,V, C, N
                                    #print("inp : " + str(inp.shape))
                                    inp = torch.cat((inp, vis[..., view_list]), dim = -2)
                                    B = inp.shape[0]
                                    if cfg.TEST.TRIANGULATE:
                                        trj_3d = HumanCam.p2d_cam3d(inp[:, pad_t:pad_t+1, :,:2, :], valid_subject, view_list)#B, T, J, 3, N)
                                        loss = 0 
                                        for idx, view_idx in enumerate(view_list):
                                            loss_view_tmp = eval_metrc(cfg, trj_3d[..., idx], inputs_3d_gt[..., view_idx])
                                            loss += loss_view_tmp.item()
                                            action_mpjpe[valid_subject][num_view - 1][view_idx] += loss_view_tmp.item() * inputs_3d_gt.shape[0]

                                        action_mpjpe[valid_subject][num_view - 1][-1] += loss * inputs_3d_gt.shape[0]
                                        continue

                                    """inp_flip = copy.deepcopy(inp)
                                    inp_flip[:,:,:,0] *= -1
                                    inp_flip[:,:,joints_left + joints_right] = inp_flip[:,:,joints_right + joints_left]"""

                                    if cfg.NETWORK.USE_GT_TRANSFORM:
                                        rotation = get_rotation(inputs_3d_gt[..., view_list])
                                        rotation = rotation.repeat(1, 1, 1, inp.shape[1], 1, 1)
                                    else:
                                        rotation = None
                                    out, other_info = model_test(inp, rotation)
                                    #out[:,:,0] = 0
                                    #print("out : " + str(out.shape))
                                    out = out.detach()#.cpu()
                                    """if EVAL and args.vis_3d and cfg.DATA.HAS_2D_PREDS:
                                        vis_tool.show(inputs_2d_pre[:,pad_t], out[:,0], inputs_3d_gt[:,0])"""

                                    """if cfg.TEST.TEST_ROTATION:
                                        out = test_multi_view_aug(out, vis[...,view_list])
                                        out[:,:,0] = 0"""
                                    
                                    """if cfg.NETWORK.USE_GT_TRANSFORM and EVAL and len(view_list) > 1 and cfg.TEST.ALIGN_TRJ:
                                        #TODO: 使用T帧姿态进行三角剖分得到平均骨骼长度再对齐
                                        trj_3d = HumanCam.p2d_cam3d(inp[:, pad_t:pad_t+1, :,:2, :], valid_subject, view_list)#B, T, J, 3, N)
                                        out_align = align_target_numpy(cfg, out, trj_3d)
                                        out_align[:,:,0] = 0
                                        out = out_align"""
                                        
                                    loss = 0
                                    pos_gt = inputs_3d_gt[..., view_list]
                                    #print("out.shape : " + str(out.shape))
                                    out = out.permute(0, 1, 4, 2, 3).contiguous()  # (B, T, N, J. C)
                                    pos_gt = pos_gt.permute(0, 1, 4, 2, 3).contiguous()
                                    #print(pos_gt.shape)
                                    points_3d_world = pos_gt[:, :, :, :, point_indexes]
                                    init_shape = points_3d_world.shape
                                    #print(points_3d_world)
                                    points_3d_world = torch.cat((points_3d_world, torch.ones(points_3d_world.shape[:-1] + (1,)).cuda()),axis=-1)
                                    #print(points_3d_world.shape)
                                    points_3d_world = torch.permute(points_3d_world.view(*points_3d_world.shape[:3], -1, 4), (0, 1, 3, 2, 4)).contiguous()
                                    #print(points_3d_world.shape)
                                    point_3D_cam = torch.einsum('voi, btpvi->btpvo', extrinsics, points_3d_world)
                                    #print(point_3D_cam.shape)
                                    point_3D_cam = torch.permute(point_3D_cam, (0, 1, 3, 2, 4)).contiguous()
                                    #print(point_3D_cam.shape)
                                    point_3D_cam = point_3D_cam.view(*init_shape[:-1], 4)
                                    point_3D_cam = point_3D_cam.view(*init_shape[:-1], 4)
                                    #point_3D_cam = (point_3D_cam[:, :, :, :, :,:3] - min_3D_cam_720_ex) / range_3D_cam_720_ex
                                    point_3D_cam = point_3D_cam[:, :, :, :, :, :3]

                                    x_gt_obj_center = torch.mean(point_3D_cam[:, :, :, :, :, 0], dim=-1, keepdim=True)
                                    y_gt_obj_center = torch.mean(point_3D_cam[:, :, :, :, :, 1], dim=-1, keepdim=True)
                                    z_gt_obj_center = torch.mean(point_3D_cam[:, :, :, :, :, 2], dim=-1, keepdim=True)
                                    point_3D_cam = point_3D_cam[:, :, :, :, :, :3]
                                    # print(1)
                                    gt_obj_center = torch.cat((x_gt_obj_center, y_gt_obj_center, z_gt_obj_center), -1)
                                    gt_obj_center = torch.unsqueeze(gt_obj_center, -2)

                                    pred_center = torch.unsqueeze(out[:, :, :, :, [6, 7, 8]],-2) * range_gt_obj_center_720_ex + min_gt_obj_center_720_ex
                                    pred_shift_center = out[:, :, :, :,point_indexes] * range_gt_shifted_obj_720_ex + min_gt_shifted_obj_720_ex
                                    out_points = pred_shift_center + pred_center
                                    #print("out.shape : "+str(out.shape))
                                    #print("point_3D_cam.shape : "+str(point_3D_cam.shape))
                                    #print("out[0] : " + str(out[0]))
                                    #out_points = out[:, :, :, :, point_indexes]*range_3D_cam_720_ex+min_3D_cam_720_ex
                                    if EVAL:
                                        x = pred_center[0, 0, 0, :, :, 0].detach().cpu().numpy()
                                        y = pred_center[0, 0, 0, :, :, 1].detach().cpu().numpy()
                                        z = pred_center[0, 0, 0, :, :, 2].detach().cpu().numpy()
                                        fig = plt.figure("visu")
                                        ax = plt.axes(projection='3d')
                                        # ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
                                        ax.scatter3D(x, y, z, marker='o', c='r')

                                        x = out_points[0, 0, 0, :, :, 0].detach().cpu().numpy()
                                        y = out_points[0, 0, 0, :, :, 1].detach().cpu().numpy()
                                        z = out_points[0, 0, 0, :, :, 2].detach().cpu().numpy()
                                        fig = plt.figure("visu")
                                        ax = plt.axes(projection='3d')
                                        #ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
                                        ax.scatter3D(x, y, z, marker='o')
                                        print(point_3D_cam[0, 0, 0, :, :, :])
                                        #plt.show()
                                        x = point_3D_cam[0, 0, 0, :, :, 0].detach().cpu().numpy()
                                        y = point_3D_cam[0, 0, 0, :, :, 1].detach().cpu().numpy()
                                        z = point_3D_cam[0, 0, 0, :, :, 2].detach().cpu().numpy()
                                        #fig = plt.figure("gt points")
                                        #ax = plt.axes(projection='3d')
                                        #ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
                                        ax.scatter3D(x, y, z, marker='^')
                                        plt.show()

                                    #print("out_points[0] : "+str(out_points[0]))
                                    for idx, view_idx in enumerate(view_list):
                                        # loss = mpjpe(out , pos_gt[:,pad:pad+1])
                                        #loss = mpjpe(out[:, :, :, :, point_indexes], point_3D_cam[:, :, :, :, :, :3])
                                        #print(point_3D_cam.shape)
                                        #print(out_points.shape)
                                        loss = mpjpe(out_points, point_3D_cam)
                                        #loss_view_tmp = eval_metrc(cfg, out[..., idx], inputs_3d_gt[..., view_idx])
                                        loss_view_tmp = eval_metrc(cfg, out_points[:, :, idx], point_3D_cam[:, :, idx, :, :, :])
                                        #cfg.TEST.NUM_FRAMES, cfg.OBJ_POSE_DATA.SUBJECTS_TRAIN, cfg.TEST.NUM_VIEWS, NUM_VIEW
                                        eval_results[t_len_id, valid_subject_id, num_view_id, view_list_id, idx] += loss_view_tmp
                                        #print("cam idx "+str(idx)+" "+str(eval_results[t_len_id, valid_subject_id, num_view_id, view_list_id, :]))
                                        #input()
                                        loss += loss_view_tmp.item()
                                        action_mpjpe[valid_subject][num_view - 1][view_idx] += loss_view_tmp.item() * inputs_3d_gt.shape[0]
                                    n_batch +=1
                                    #print("n_batch : "+str(n_batch))
                                    action_mpjpe[valid_subject][num_view - 1][-1] += loss * inputs_3d_gt.shape[0]
                            eval_results[t_len_id, valid_subject_id, num_view_id, view_list_id, :] /= n_batch
                            """print("final n_batch : " + str(n_batch))
                            print(eval_results[t_len_id, valid_subject_id, num_view_id, view_list_id, :])
                            input()"""
                            valid_subject_id += 1
                        view_list_id += 1
                    num_view_id+=1
                t_len_id += 1
                np.save("log/eval_epoch_"+str(epoch)+"_t_len_"+str(t_len)+".npy",eval_results)
                print('num_actions :{}'.format(len(action_frames)))
                for task_idx in range(len(cfg.OBJ_POSE_DATA.SUBJECTS_TEST)):
                    tsk = cfg.OBJ_POSE_DATA.SUBJECTS_TEST[task_idx]
                    for i in range(NUM_VIEW):
                        summary_writer.add_scalar("test_mpjpe_tsk{}_view{}/epoch".format(tsk, i), eval_results[0, task_idx, 0, 0, i], epoch)

                for num_view in cfg.TEST.NUM_VIEWS:
                    tmp = [0] * (NUM_VIEW + 1)
                    print('num_view:{}'.format(num_view))
                    for act in action_mpjpe:
                        for i in range(NUM_VIEW):
                            action_mpjpe[act][num_view - 1][i] /= (action_frames[act] * N[num_view - 1][i])
                        action_mpjpe[act][num_view - 1][-1] /= (action_frames[act] * N[num_view - 1][-1] * num_view)
                        print('mpjpe of {:18}'.format(act), end = ' ')
                        for i in range(NUM_VIEW):
                            print('view_{}: {:.3f}'.format(cfg.H36M_DATA.TEST_CAMERAS[i], action_mpjpe[act][num_view - 1][i] * 1000), end = '    ')
                            tmp[i] += action_mpjpe[act][num_view - 1][i] * 1000

                        print('avg_action: {:.3f}'.format(action_mpjpe[act][num_view - 1][-1] * 1000))
                        tmp[-1] += action_mpjpe[act][num_view - 1][-1] * 1000
                    print('avg:', end = '                        ')
                    for i in range(NUM_VIEW):
                        print('view_{}: {:.3f}'.format(i, tmp[i] / len(action_frames)), end = '    ')
                    print('avg_all   : {:.3f}'.format(tmp[-1] / len(action_frames)))
                        
                    if summary_writer is not None:
                        summary_writer.add_scalar("test_mpjpe_t{}_v{}/epoch".format(t_len, num_view), tmp[-1] / len(action_frames), epoch)
                    epoch_loss_valid += tmp[-1] / len(action_frames)
                epoch_loss_valid /= len(cfg.TEST.NUM_VIEWS)
                print('t_len:{} avg:{:.3f}'.format(t_len, epoch_loss_valid))
                
            if EVAL:
                eval_elapsed = (time.time() - eval_start_time)/60
                print('time:{:.2f}'.format(eval_elapsed))
                exit()
            
            
        epoch += 1
        
        if epoch_loss_valid < best_result:
            best_result = epoch_loss_valid
            best_state_dict = copy.deepcopy(model.module.state_dict())
            best_result_epoch = epoch
        elapsed = (time.time() - start_time)/60
        print('epoch:{:3} time:{:.2f} lr:{:.9f} best_result_epoch:{:3} best_result:{:.3f}'.format(epoch, elapsed, lr, best_result_epoch, best_result))
        print('checkpoint:{}'.format(cfg.TRAIN.CHECKPOINT))
        # Decay learning rate exponentially
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        momentum = initial_momentum * np.exp(-epoch/cfg.TRAIN.NUM_EPOCHES * np.log(initial_momentum/final_momentum))
        model.module.set_bn_momentum(momentum)
            
        torch.save({
                'epoch': epoch,
                'lr': lr,
                'random_state': train_generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model':model.module.state_dict(),
                'best_epoch': best_result_epoch,
                'best_model':best_state_dict,
            }, cfg.TRAIN.CHECKPOINT)
          


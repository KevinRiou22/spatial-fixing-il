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

if args.test_success_rate:
    keypoints_boxes_2d = dict(np.load(cfg.OBJ_POSE_DATA.ROOT_DIR + '/sim_all_formal_box2d_task_test.npz', allow_pickle=True))['pose'].item()
    keypoints_waypoints = dict(np.load(cfg.OBJ_POSE_DATA.ROOT_DIR + '/sim_all_formal_waypoints_task_test.npz', allow_pickle=True))['pose'].item()
else:
    keypoints_boxes_2d = dict(np.load(cfg.OBJ_POSE_DATA.ROOT_DIR + '/sim_all_formal_box2d_1_frame.npz', allow_pickle=True))['pose'].item()
    keypoints_waypoints = dict(np.load(cfg.OBJ_POSE_DATA.ROOT_DIR+'/sim_all_formal_waypoints_1_frame.npz', allow_pickle=True))['pose'].item()

action_frames = {}
for act in cfg.OBJ_POSE_DATA.SUBJECTS_TRAIN:
    action_frames[act] = 0
n_ex=0
for index_tsk, (tsk, exemples) in enumerate(keypoints_waypoints.items()):
    if tsk in cfg.OBJ_POSE_DATA.SUBJECTS_TRAIN:
        for index_ex, (ex, traj) in enumerate(exemples.items()):
            #N_frame_action_dict[traj[0].shape[0]] = tsk
            action_frames[tsk] += traj[0].shape[0]
            n_ex += 1

actions = []
train_actions = []
test_actions = []
vis_actions = []
#n_ex=50 if args.test_success_rate else 300
for k in range(n_ex):
    ex = "exemple_"+str(k)
    actions.append(ex)
    if args.test_success_rate:
        train_actions.append(ex)
        test_actions.append(ex)
        vis_actions.append(ex)
    else:
        if k<int(n_ex*0.8):
            train_actions.append(ex)
        else:
            test_actions.append(ex)
            vis_actions.append(ex)

def fetch(subjects, action_filter=None,  parse_3d_poses=True, is_test = False, fetch_what="waypoints"):
    out_poses_3d = []
    out_poses_2d_view1 = []
    out_poses_2d_view2 = []
    out_poses_2d_view3 = []
    out_poses_2d_view4 = []
    out_camera_params = []
    out_subject_action = []
    used_cameras = cfg.H36M_DATA.TEST_CAMERAS if is_test else cfg.H36M_DATA.TRAIN_CAMERAS
    #print(keypoints.keys())
    if fetch_what=="boxes":
        to_fetch=keypoints_boxes_2d
    else:
        to_fetch=keypoints_waypoints
    for subject in subjects:
        for action in to_fetch[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action==a:
                        found = True
                        break
                if not found:
                    continue
                
            poses_2d = to_fetch[subject][action]

            out_subject_action.append([subject, action])
            n_frames = poses_2d[0].shape[0]
            if fetch_what=="boxes":
                out_poses_2d_view1.append(poses_2d[0])
                out_poses_2d_view2.append(poses_2d[1])
                out_poses_2d_view3.append(poses_2d[2])
                out_poses_2d_view4.append(poses_2d[3])
            elif fetch_what=="waypoints":
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

def q_conjugate(q):
    q[..., 1:]= -q[..., 1:]
    return q

def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.array([w, x, y, z])

def q_mult_batch(q1, q2):
    w1, x1, y1, z1 = q1[..., 0:1], q1[..., 1:2], q1[..., 2:3], q1[..., 3:4]
    w2, x2, y2, z2 = q2[..., 0:1], q2[..., 1:2], q2[..., 2:3], q2[..., 3:4]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return torch.cat((w, x, y, z), -1)

def qv_mult(q1, v1):
    q2 = np.concatenate([np.array([0.0]), v1])
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]

def qv_mult_batch(q1, v1):
    zeros = torch.zeros(*v1.shape[:-1], 1).to(q1.device)
    q2 = torch.cat((zeros, v1), dim=-1)
    return q_mult_batch(q_mult_batch(q1, q2), q_conjugate(q1))[..., 1:]


if True:
    if not cfg.DEBUG and (not args.eval or args.log):
        summary_writer = SummaryWriter(log_dir=cfg.LOG_DIR)
    else:
        summary_writer = None
    
    imgs_train_2d, subject_action = fetch(cfg.OBJ_POSE_DATA.SUBJECTS_TRAIN, train_actions, fetch_what="boxes")
    wayponts_train_2d, subject_action = fetch(cfg.OBJ_POSE_DATA.SUBJECTS_TRAIN, train_actions, fetch_what="waypoints")
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
        lr = 0.0005#checkpoint['lr']
        best_result = 100
    else:
        epoch = 0
        best_result = 100
        best_state_dict = None
        best_result_epoch = 0
        
    lr_decay = cfg.TRAIN.LR_DECAY
    initial_momentum = 0.1
    final_momentum = 0.001

    train_generator = ChunkedGenerator(cfg.TRAIN.BATCH_SIZE, imgs_train_2d, wayponts_train_2d, 1,pad=pad, causal_shift=causal_shift, shuffle=True, augment=False, sub_act=subject_action, images_path=cfg.OBJ_POSE_DATA.IMAGES_PATH) if cfg.H36M_DATA.PROJ_Frm_3DCAM == True else ChunkedGenerator(cfg.TRAIN.BATCH_SIZE, imgs_train_2d, wayponts_train_2d, 1,pad=pad, causal_shift=causal_shift, shuffle=True, augment=False, images_path=cfg.OBJ_POSE_DATA.IMAGES_PATH)
 
    print('** Starting.')
    
    data_aug = DataAug(cfg, add_view = cfg.TRAIN.NUM_AUGMENT_VIEWS)
    iters = 0
    msefun = torch.nn.L1Loss() 
    num_train_views = len(cfg.H36M_DATA.TRAIN_CAMERAS) + cfg.TRAIN.NUM_AUGMENT_VIEWS


    while epoch < cfg.TRAIN.NUM_EPOCHES:
        start_time = time.time()
        model.train()
        process = tqdm(total = train_generator.num_batches)
        #deb_idx=0
        for batch_2d_boxes, batch_2d_waypoints, sub_action in train_generator.next_epoch():
            if EVAL:
                break
            process.update(1)
            inputs = torch.from_numpy(batch_2d_boxes.astype('float32'))
            labels = torch.from_numpy(batch_2d_waypoints.astype('float32'))
            if cfg.DATA.HAS_2D_PREDS:
                inputs_2d_pre = (inputs/256).permute(0,1,4,2,3,5).contiguous()

                cam_3d = labels[..., 4:12, :]#inputs[..., 2*cfg.NETWORK.INPUT_DIM:dims_sum, :]
                labels_2d = labels[..., :2, :]/256
            else:
                inputs_2d_gt = inputs

            inputs_3d_gt = cam_3d.cuda()
            view_list = list(range(num_train_views))
    

            if use_2d_gt or (not cfg.DATA.HAS_2D_PREDS):
                h36_inp = inputs_2d_gt[..., view_list]
            else:
                h36_inp = inputs_2d_pre[..., view_list]
            pos_gt = inputs_3d_gt[..., view_list]

            p3d_gt_ori = copy.deepcopy(pos_gt)
            p3d_root = copy.deepcopy(pos_gt[:,:,:1])
            optimizer.zero_grad()
            inp = h36_inp#torch.cat((h36_inp, vis), dim = -2)

            rotation = None
 
            if cfg.TRAIN.USE_INTER_LOSS:
                print('Input shape is {}'.format(inp.shape))
                out, keypoints_2D, other_out, tran, pred_rot = model(inp, rotation) #mask:(B, 1, 1, 1, N, N)
            else:
                out, keypoints_2D = model(inp, rotation)


            out = out.permute(0, 1, 4, 2, 3).contiguous()  # (B, T, N, J. C)
            pos_gt = pos_gt.permute(0, 1, 4, 2, 3).contiguous()
            pos_gt = torch.max(pos_gt, 2, keepdim=True)[0].repeat(1, 1, 4, 1, 1)
            loss_pose = mpjpe(out[:, :, :, :, :3], pos_gt[:, pad:pad + 1, :, :, :3])
            loss_quat = mpjpe(out[:, :, :, :, 3:7], pos_gt[:, pad:pad + 1, :, :, 3:7])
            loss_open = mpjpe(out[:, :, :, :, 7:], pos_gt[:, pad:pad + 1, :, :, 7:])
            loss_2d = mpjpe(labels_2d.permute(0,1,2,4,3).contiguous().to(out.device), keypoints_2D.permute(0,1,2,4,3).contiguous())
            print("loss_pose : "+str(loss_pose))
            print("loss_quat : "+str(loss_quat))
            print("loss_open : "+str(loss_open))
            print("loss_2d : "+str(loss_2d))
            loss = 10*loss_pose+loss_quat+loss_open#+loss_2d

            if summary_writer is not None:
                summary_writer.add_scalar("loss_pose/iter", loss_pose, iters)
                summary_writer.add_scalar("loss_quat/iter", loss_quat, iters)
                summary_writer.add_scalar("loss_open/iter", loss_open, iters)
                summary_writer.add_scalar("loss_2d/iter", loss_2d, iters)
                summary_writer.add_scalar("loss_final/iter", loss, iters)

            if pred_rot is not None and cfg.TRAIN.USE_ROT_LOSS:
                print("rot loss : ")
                tran_loss = msefun(pred_rot, rotation)

                if summary_writer is not None:
                    summary_writer.add_scalar("loss_tran/iter", tran_loss, iters)
                loss = loss + cfg.TRAIN.ROT_LOSS_WEIGHT * tran_loss

            loss_consis_weight = cfg.TRAIN.CONSIS_LOSS_WEIGHT
            loss_consis = 0
            print('Supervised Loss is {}'.format(loss))
            loss_total = loss
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
            # eval_results = np.zeros((len(cfg.TEST.NUM_FRAMES), len(cfg.OBJ_POSE_DATA.SUBJECTS_TRAIN), len(cfg.TEST.NUM_VIEWS), NUM_VIEW, 4))
            eval_results = []
            t_len_id = 0
            for t_len in cfg.TEST.NUM_FRAMES:
                eval_results.append([])
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
                    eval_results[-1].append([])
                    pad_t = t_len // 2
                    view_list_id = 0
                    for view_list in itertools.combinations(list(range(NUM_VIEW)), num_view):
                        eval_results[-1][-1].append([])
                        view_list = list(view_list)
                        N[num_view - 1][-1] += 1
                        for i in view_list:
                            N[num_view - 1][i] += 1
                        valid_subject_id = 0
                        for valid_subject in cfg.OBJ_POSE_DATA.SUBJECTS_TEST:
                            eval_results[-1][-1][-1].append([[[], [], [], [], []], [[], [], [], [], []], [[], [], [], [], []], [[], [], [], [], []]])
                            n_batch = 0
                            # eval_results[-1][-1][-1][-1].append([[], [], [], []])
                            for act in vis_actions if EVAL else test_actions:
                                boxes_valid_2d = fetch([valid_subject], [act], is_test =True, fetch_what="boxes")
                                waypoints_valid_2d = fetch([valid_subject], [act], is_test =True, fetch_what="waypoints")
                                print(valid_subject)
                                print(act)
                                print(boxes_valid_2d)
                                #print("poses_valid_2d : "+str(len(poses_valid_2d)))
                                test_generator = ChunkedGenerator(cfg.TEST.BATCH_SIZE, boxes_valid_2d, waypoints_valid_2d, 1,pad=pad_t, causal_shift=causal_shift, shuffle=False, augment=False, images_path=cfg.OBJ_POSE_DATA.IMAGES_PATH)
                                #batch_eval_results = np.zeros((len(view_list), ))
                                ts_counter=0
                                for batch_2d_boxes, batch_waypoints, _ in test_generator.next_epoch():
                                    inputs = torch.from_numpy(batch_2d_boxes.astype('float32')).cuda()
                                    labels = torch.from_numpy(batch_waypoints.astype('float32')).cuda()
                                    if cfg.DATA.HAS_2D_PREDS:
                                        # inputs_2d_gt = inputs[..., :, :cfg.NETWORK.INPUT_DIM, :]/256
                                        inputs_2d_pre = (inputs / 256).permute(0, 1, 4, 2, 3, 5).contiguous()
                                        labels_2d = labels[..., :2, :] / 256
                                        cam_3d = labels[..., 4:12, :]  # inputs[..., 2*cfg.NETWORK.INPUT_DIM:dims_sum, :]
                                    else:
                                        inputs_2d_gt = inputs
                                        cam_3d = labels

                                    vis = inputs[...,-1,:].cuda()
                                    inputs_3d_gt = cam_3d[:,pad_t:pad_t+1]

                                    inp = inputs_2d_pre
                                    B = inp.shape[0]

                                    rotation = None
                                    print(inp.shape)
                                    out, keypoints_2D, other_info = model_test(inp, rotation)
                                    # out[:,:,0] = 0
                                    # print("out : " + str(out.shape))
                                    out = out.detach()  # .cpu()
                                    if args.test_success_rate:
                                        np.save("./log/"+str(act)+"_ts_"+str(ts_counter)+".npy", out.cpu().numpy())
                                    ts_counter+=1
                                    inputs_3d_gt = torch.max(inputs_3d_gt, -1, keepdim=True)[0].repeat(1, 1, 1, 1, 4)

                                    loss = 0
                                    # print(inputs_3d_gt.shape)
                                    pos_gt = inputs_3d_gt[..., view_list]

                                    # print("out_points[0] : "+str(out_points[0]))
                                    for idx, view_idx in enumerate(view_list):
                                        loss = mpjpe(out, pos_gt)
                                        # loss = mpjpe(out[:, :, :, :, point_indexes], point_3D_cam[:, :, :, :, :, :3])
                                        # print(point_3D_cam.shape)
                                        # print(out_points.shape)
                                        # loss = mpjpe(out_points, point_3D_cam)
                                        loss_pose = mpjpe(out[:, :, :, :3, idx], pos_gt[:, pad:pad + 1, :,  :3, idx])
                                        loss_quat = mpjpe(out[:, :, :,  3:7, idx], pos_gt[:, pad:pad + 1, :,  3:7, idx])
                                        loss_open = mpjpe(out[:, :, :, 7:, idx], pos_gt[:, pad:pad + 1, :,  7:, idx])

                                        loss_2d = mpjpe(labels_2d[..., idx].to(out.device), keypoints_2D[..., idx])

                                        loss_view_tmp = eval_metrc(cfg, out[..., idx], inputs_3d_gt[:, :, :, :, view_idx])

                                        # loss_view_tmp = eval_metrc(cfg, out_points[:, :, idx], point_3D_cam[:, :, idx, :, :, :])
                                        # cfg.TEST.NUM_FRAMES, cfg.OBJ_POSE_DATA.SUBJECTS_TRAIN, cfg.TEST.NUM_VIEWS, NUM_VIEW
                                        # eval_results[t_len_id, valid_subject_id, num_view_id, view_list_id, idx] += loss_view_tmp

                                        eval_results[t_len_id][num_view_id][view_list_id][valid_subject_id][idx][0].append(loss_pose.cpu())
                                        eval_results[t_len_id][num_view_id][view_list_id][valid_subject_id][idx][1].append(loss_quat.cpu())
                                        eval_results[t_len_id][num_view_id][view_list_id][valid_subject_id][idx][2].append(loss_open.cpu())
                                        eval_results[t_len_id][num_view_id][view_list_id][valid_subject_id][idx][3].append(loss_2d.cpu())
                                        eval_results[t_len_id][num_view_id][view_list_id][valid_subject_id][idx][4].append(loss_view_tmp.cpu())

                                        # print("cam idx "+str(idx)+" "+str(eval_results[t_len_id, valid_subject_id, num_view_id, view_list_id, :]))
                                        # input()
                                        loss += loss_view_tmp.item()
                                        action_mpjpe[valid_subject][num_view - 1][view_idx] += loss_view_tmp.item() * inputs_3d_gt.shape[0]
                                    n_batch += 1
                                    # print("n_batch : "+str(n_batch))
                                    action_mpjpe[valid_subject][num_view - 1][-1] += loss * inputs_3d_gt.shape[0]
                                    # eval_results[t_len_id, valid_subject_id, num_view_id, view_list_id, :] /= n_batch
                            valid_subject_id += 1
                        view_list_id += 1


                        #np.save("log/eval_epoch_" + str(epoch) + "_t_len_" + str(t_len) + ".npy", eval_results)
                        print('num_actions :{}'.format(len(action_frames)))
                        for task_idx in range(len(cfg.OBJ_POSE_DATA.SUBJECTS_TEST)):
                            tsk = cfg.OBJ_POSE_DATA.SUBJECTS_TEST[task_idx]
                            for i in range(NUM_VIEW):
                                l_id = 0
                                for l in ["l_pose", "l_quat", "l_grip", "l_2d", "l_tot"]:
                                    view_mean_res_val = np.mean(np.array(eval_results[0][0][0][task_idx][i][l_id]))
                                    summary_writer.add_scalar("test_mpjpe_tsk{}_view{}_{}/epoch".format(tsk, i, l), view_mean_res_val, epoch)
                                    l_id+=1

                        for num_view in cfg.TEST.NUM_VIEWS:
                            tmp = [0] * (NUM_VIEW + 1)
                            print('num_view:{}'.format(num_view))
                            for act in action_mpjpe:
                                for i in range(NUM_VIEW):
                                    action_mpjpe[act][num_view - 1][i] /= (action_frames[act] * N[num_view - 1][i])
                                action_mpjpe[act][num_view - 1][-1] /= (
                                            action_frames[act] * N[num_view - 1][-1] * num_view)
                                print('mpjpe of {:18}'.format(act), end=' ')
                                for i in range(NUM_VIEW):
                                    print('view_{}: {:.3f}'.format(cfg.H36M_DATA.TEST_CAMERAS[i],
                                                                   action_mpjpe[act][num_view - 1][i] * 1000),
                                          end='    ')
                                    tmp[i] += action_mpjpe[act][num_view - 1][i] * 1000

                                print('avg_action: {:.3f}'.format(action_mpjpe[act][num_view - 1][-1] * 1000))
                                tmp[-1] += action_mpjpe[act][num_view - 1][-1] * 1000
                            print('avg:', end='                        ')
                            for i in range(NUM_VIEW):
                                print('view_{}: {:.3f}'.format(i, tmp[i] / len(action_frames)), end='    ')
                            print('avg_all   : {:.3f}'.format(tmp[-1] / len(action_frames)))

                            if summary_writer is not None:
                                summary_writer.add_scalar("test_mpjpe_t{}_v{}/epoch".format(t_len, num_view),
                                                          tmp[-1] / len(action_frames), epoch)
                            epoch_loss_valid += tmp[-1] / len(action_frames)
                        epoch_loss_valid /= len(cfg.TEST.NUM_VIEWS)
                        print('t_len:{} avg:{:.3f}'.format(t_len, epoch_loss_valid))
                    num_view_id += 1

                    if EVAL:
                        eval_elapsed = (time.time() - eval_start_time) / 60
                        print('time:{:.2f}'.format(eval_elapsed))
                        exit()

                    epoch += 1
                t_len_id += 1
        
        if epoch_loss_valid < best_result:
            best_result = epoch_loss_valid
            best_state_dict = copy.deepcopy(model.module.state_dict())
            best_result_epoch = epoch
        elapsed = (time.time() - start_time)/60
        print('epoch:{:3} time:{:.2f} lr:{:.9f} best_result_epoch:{:3} best_result:{:.3f}'.format(epoch, elapsed, lr, best_result_epoch, best_result))
        print('checkpoint:{}'.format(cfg.TRAIN.CHECKPOINT))
        # Decay learning rate exponentially
        #if epoch>=60:
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
          


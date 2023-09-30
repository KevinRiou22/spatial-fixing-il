from itertools import zip_longest
import numpy as np
import torch
from torch.utils.data import Dataset
import sys,os
import copy
import random
import pickle
from common.camera import *
from common.set_seed import *
import itertools
import cv2
set_seed()
this_dir = os.path.dirname(__file__)
sys.path.insert(0, this_dir)
 
class ChunkedGenerator(Dataset):
    def __init__(self, batch_size, boxes_train_2d, waypoints_train_2d,
                 chunk_length, camera_param = None, pad=0, causal_shift=0,
                 shuffle=True, random_seed=1234,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False,  step = 1, sub_act = None, images_path=''):

        tmp_boxes = []
        tmp_waypoints = []
        num_cam = len(boxes_train_2d)  #len(poses_2d)=4; len(poses_2d[0])=150
        #len(sub_act) = 150
        self.VIEWS = range(num_cam)
        self.images_path = images_path
        for i in range(len(boxes_train_2d[0])): #num of videos
            n_frames = 10000000000
            for n in range(num_cam): #num of cams
                if boxes_train_2d[n][i].shape[0] < n_frames:
                    n_frames = boxes_train_2d[n][i].shape[0]
                
            for n in range(num_cam):
                boxes_train_2d[n][i] = boxes_train_2d[n][i][:n_frames]
                waypoints_train_2d[n][i] = waypoints_train_2d[n][i][:n_frames]
            temp_pos_boxes = boxes_train_2d[0][i][..., np.newaxis]  #第0个view的第i段视频
            temp_pos_waypoints = waypoints_train_2d[0][i][..., np.newaxis]
            for j in range(1, num_cam):
                temp_pos_boxes = np.concatenate((temp_pos_boxes, boxes_train_2d[j][i][...,np.newaxis]), axis = -1)
                temp_pos_waypoints = np.concatenate((temp_pos_waypoints, waypoints_train_2d[j][i][...,np.newaxis]), axis = -1)

            tmp_boxes.append(temp_pos_boxes)
            tmp_waypoints.append(temp_pos_waypoints)
        self.db_boxes = tmp_boxes  #len(self.db)=150; len(self.db[0])=2478; len(self.db[0][0])=17; len(self.db[0][0][0])=8; len(self.db[0][0][0][0])=4
        self.db_waypoints = tmp_waypoints
        # print("(len(self.db_waypoints):"+str((len(self.db_waypoints))))
        # print("(len(self.db_waypoints[0]):" + str((len(self.db_waypoints[0]))))
        # print("(len(self.db_boxes):" + str((len(self.db_boxes))))
        # print("(len(self.db_boxes[0]):" + str((len(self.db_boxes[0]))))
        self.sub_act = sub_act
        # Build lineage info
        pairs = [] # (seq_idx, start_frame, end_frame, flip) tuples

        for i in range(len(boxes_train_2d[0])):#num of videos
            n_chunks = (boxes_train_2d[0][i].shape[0] + chunk_length - 1) // chunk_length
            sub_act_crt = self.sub_act[i] if self.sub_act is not None else None
            offset = (n_chunks * chunk_length - boxes_train_2d[0][i].shape[0]) // 2
            bounds = np.arange(n_chunks+1)*chunk_length - offset
            augment_vector = np.full(len(bounds - 1), False, dtype=bool)

            if sub_act_crt is not None:
                pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], augment_vector, [tuple(sub_act_crt)]*len(bounds - 1))
            else:
                pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], augment_vector)
            if augment:
                if sub_act_crt is not None:
                    pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], ~augment_vector, [tuple(sub_act_crt)]*len(bounds - 1))
                else:
                    pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], ~augment_vector)
        pairs = pairs[::step]
        """print("len(pairs) : " + str(len(pairs)))
        print("len(pairs[0]) : " + str(len(pairs[0])))

        print("pairs[20][0] : "+str(pairs[20][0]))
        print("pairs[20][1] : " + str(pairs[20][1]))
        print("pairs[20][2] : " + str(pairs[20][2]))
        print("pairs[20][3] : " + str(pairs[20][3]))
        input()"""
        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None

        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right

        self.batch_2d_boxes = np.empty((batch_size, chunk_length + 2*pad, 256, 256, 3, num_cam))

        self.batch_2d_waypoints = np.empty((batch_size, chunk_length + 2 * pad, waypoints_train_2d[0][0].shape[-2], waypoints_train_2d[0][0].shape[-1], num_cam))
        #(B, 7, 17, 8, 4)
        self.label_sub_act = np.empty((batch_size,)).tolist()
                
    def num_frames(self):
        return self.num_batches * self.batch_size
    
    def random_state(self):
        return self.random
    
    def set_random_state(self, random):
        self.random = random
        
    def augment_enabled(self):
        return self.augment
    
    def next_pairs(self):
        if self.state is None:
            #print('***************************************************')
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state

    def next_epoch(self):
        #path_dataset = "/media/facebooksurrounduser/data/RLBench_data/multi_view_2D_3D_keypoints_300ex_paper/"
        #path_dataset = "/home/facebooksurrounduser/Documents/RLBench_works/pose_obj_3d_from_image/data/RLBench_pose_est_data/"
        #path_dataset = "/media/facebooksurrounduser/data/RLBench_data/test_set/train/"
        enabled = True
        while enabled:
            start_idx, pairs = self.next_pairs()
            for b_i in range(start_idx, self.num_batches):
                chunks = pairs[b_i*self.batch_size : (b_i+1)*self.batch_size] #(B, 4) [vid_inx, frm_inx_str, frm_inx_ed, flip]
                if self.sub_act is None:
                    for i, (seq_i, start_3d, end_3d, flip) in enumerate(chunks):
                        start_2d = start_3d - self.pad - self.causal_shift
                        end_2d = end_3d + self.pad - self.causal_shift
                        # 2D poses
                        im_idx = self.db_boxes[seq_i]


                        #print(seq_2d_boxes.shape)
                        #input()
                        #seq_2d_boxes = self.db_boxes[seq_i]
                        seq_2d_waypoints = self.db_waypoints[seq_i]
                        low_2d = max(start_2d, 0)
                        high_2d = min(end_2d, im_idx.shape[0])
                        pad_left_2d = low_2d - start_2d
                        pad_right_2d = end_2d - high_2d

                        if pad_left_2d != 0 or pad_right_2d != 0:
                            seq_2d_boxes = []
                            for im_data_pt in im_idx[low_2d:high_2d]:
                                mode = "train"

                                seq_2d_boxes.append([])
                                c_id = 0
                                for cam in ["left_shoulder", "right_shoulder", "overhead", "front"]:
                                    tsk, var, ex, t = im_data_pt[:, c_id].astype(int)
                                    # print(tsk, var, ex, t)
                                    # input()
                                    img_path = self.images_path + str(tsk) + "/obs/" + str(cam) + "/" + str(
                                        var) + "/" + str(ex) + "_rgb_" + str(t) + ".png"
                                    seq_2d_boxes[-1].append(cv2.imread(img_path))
                                    c_id += 1
                            seq_2d_boxes = np.array(seq_2d_boxes)
                            seq_2d_boxes = np.transpose(seq_2d_boxes, (0, 2, 3, 4, 1))
                            #self.batch_2d_boxes[i] = np.pad(seq_2d_boxes[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0), (0, 0)), 'edge')
                            self.batch_2d_boxes[i] = np.pad(seq_2d_boxes,
                                                            ((pad_left_2d, pad_right_2d), (0, 0), (0, 0), (0, 0)),
                                                            'edge')
                            self.batch_2d_waypoints[i] = np.pad(seq_2d_waypoints[low_2d:high_2d],
                                                            ((pad_left_2d, pad_right_2d), (0, 0), (0, 0), (0, 0)),
                                                            'edge')
                        else:
                            seq_2d_boxes = []
                            for im_data_pt in im_idx[low_2d:high_2d]:
                                mode = "train"

                                seq_2d_boxes.append([])
                                c_id = 0
                                for cam in ["left_shoulder", "right_shoulder", "overhead", "front"]:
                                    tsk, var, ex, t = im_data_pt[:, c_id].astype(int)
                                    # print(tsk, var, ex, t)
                                    # input()
                                    img_path = self.images_path + str(tsk) + "/obs/" + str(cam) + "/" + str(
                                        var) + "/" + str(ex) + "_rgb_" + str(t) + ".png"
                                    seq_2d_boxes[-1].append(cv2.imread(img_path))
                                    c_id += 1
                            seq_2d_boxes = np.array(seq_2d_boxes)
                            #print(seq_2d_boxes.shape)
                            seq_2d_boxes = np.transpose(seq_2d_boxes, (0, 2, 3, 4, 1))
                            self.batch_2d_boxes[i] = seq_2d_boxes#[low_2d:high_2d]
                            self.batch_2d_waypoints[i] = seq_2d_waypoints[low_2d:high_2d]

                        # if flip:
                        #     # Flip 2D keypoints
                        #
                        #     self.batch_2d[i, :, :, 0] *= -1
                        #     self.batch_2d[i, :, self.kps_left + self.kps_right] = self.batch_2d[i, :, self.kps_right + self.kps_left]
                        #     if self.batch_2d.shape[-2] == 8:#(p2d_gt, p2d_pre, p3d, vis)
                        #         self.batch_2d[i, :, :, 2] *= -1
                        #         self.batch_2d[i, :, :, 4] *= -1
                        #     elif self.batch_2d.shape[-2] == 6:#(p2d, p3d, vis)
                        #         self.batch_2d[i, :,:,2] *= -1
                        #     elif self.batch_2d.shape[-2] == 11: #(p2d_gt, p2d_pre, p3d, trj_c3d, vis)
                        #         self.batch_2d[i, :, :, 2] *= -1
                        #         self.batch_2d[i, :, :, 4] *= -1
                        #         self.batch_2d[i, :, :, 7] *= -1
                        #     elif self.batch_2d.shape[-2] == 13: #(p2d_gt, p2d_pre, p3d, trj_c3d, trj_2d, vis)
                        #         self.batch_2d[i, :, :, 2] *= -1
                        #         self.batch_2d[i, :, :, 4] *= -1
                        #         self.batch_2d[i, :, :, 7] *= -1
                        #         self.batch_2d[i, :, :, 10] *= -1
                        #     else:
                        #         print(self.batch_2d.shape[-2])
                        #         sys.exit()
                else:
                    for i, (seq_i, start_3d, end_3d, flip, _sub_act) in enumerate(chunks):
                        start_2d = start_3d - self.pad - self.causal_shift
                        end_2d = end_3d + self.pad - self.causal_shift
                        # 2D poses
                        im_idx = self.db_boxes[seq_i]
                        seq_2d_waypoints = self.db_waypoints[seq_i]
                        self.label_sub_act[i] = _sub_act
                        low_2d = max(start_2d, 0)
                        high_2d = min(end_2d, im_idx.shape[0])
                        pad_left_2d = low_2d - start_2d
                        pad_right_2d = end_2d - high_2d

                        if pad_left_2d != 0 or pad_right_2d != 0:

                            seq_2d_boxes = []
                            for im_data_pt in im_idx[low_2d:high_2d]:
                                mode = "train"

                                seq_2d_boxes.append([])
                                c_id = 0
                                for cam in ["left_shoulder", "right_shoulder", "overhead", "front"]:
                                    tsk, var, ex, t = im_data_pt[:, c_id].astype(int)
                                    # print(tsk, var, ex, t)
                                    # input()
                                    img_path = self.images_path + str(tsk) + "/obs/" + str(cam) + "/" + str(
                                        var) + "/" + str(ex) + "_rgb_" + str(t) + ".png"
                                    seq_2d_boxes[-1].append(cv2.imread(img_path))
                                    c_id += 1
                            seq_2d_boxes = np.array(seq_2d_boxes)
                            seq_2d_boxes = np.transpose(seq_2d_boxes, (0, 2, 3, 4, 1))
                            #self.batch_2d_boxes[i] = np.pad(seq_2d_boxes[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0), (0, 0)), 'edge')
                            self.batch_2d_boxes[i] = np.pad(seq_2d_boxes,
                                                            ((pad_left_2d, pad_right_2d), (0, 0), (0, 0), (0, 0)),
                                                            'edge')
                            self.batch_2d_waypoints[i] = np.pad(seq_2d_waypoints[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0), (0, 0)), 'edge')
                        else:

                            seq_2d_boxes = []
                            for im_data_pt in im_idx[low_2d:high_2d]:
                                mode = "train"

                                seq_2d_boxes.append([])
                                c_id = 0
                                for cam in ["left_shoulder", "right_shoulder", "overhead", "front"]:
                                    tsk, var, ex, t = im_data_pt[:, c_id].astype(int)
                                    # print(tsk, var, ex, t)
                                    # input()
                                    img_path = self.images_path + str(tsk) + "/obs/" + str(cam) + "/" + str(
                                        var) + "/" + str(ex) + "_rgb_" + str(t) + ".png"
                                    seq_2d_boxes[-1].append(cv2.imread(img_path))
                                    c_id += 1
                            seq_2d_boxes = np.array(seq_2d_boxes)
                            seq_2d_boxes = np.transpose(seq_2d_boxes, (0, 2, 3, 4, 1))
                            self.batch_2d_boxes[i] = seq_2d_boxes#[low_2d:high_2d]
                            self.batch_2d_waypoints[i] = seq_2d_waypoints[low_2d:high_2d]

                        # if flip:
                        #     # Flip 2D keypoints
                        #
                        #     self.batch_2d[i, :, :, 0] *= -1
                        #     self.batch_2d[i, :, self.kps_left + self.kps_right] = self.batch_2d[i, :,
                        #                                                           self.kps_right + self.kps_left]
                        #     if self.batch_2d.shape[-2] == 8:  # (p2d_gt, p2d_pre, p3d, vis)
                        #         self.batch_2d[i, :, :, 2] *= -1
                        #         self.batch_2d[i, :, :, 4] *= -1
                        #     elif self.batch_2d.shape[-2] == 6:  # (p2d, p3d, vis)
                        #         self.batch_2d[i, :, :, 2] *= -1
                        #     elif self.batch_2d.shape[-2] == 11:  # (p2d_gt, p2d_pre, p3d, trj_c3d, vis)
                        #         self.batch_2d[i, :, :, 2] *= -1
                        #         self.batch_2d[i, :, :, 4] *= -1
                        #         self.batch_2d[i, :, :, 7] *= -1
                        #     elif self.batch_2d.shape[-2] == 13:  # (p2d_gt, p2d_pre, p3d, trj_c3d, trj_2d, vis)
                        #         self.batch_2d[i, :, :, 2] *= -1
                        #         self.batch_2d[i, :, :, 4] *= -1
                        #         self.batch_2d[i, :, :, 7] *= -1
                        #         self.batch_2d[i, :, :, 10] *= -1
                        #     else:
                        #         print(self.batch_2d.shape[-2])
                        #         sys.exit()


                if self.endless:
                    self.state = (b_i + 1, pairs)
                
                if self.sub_act is not None:
                    yield self.batch_2d_boxes[:len(chunks)], self.batch_2d_waypoints[:len(chunks)], self.label_sub_act
                else:
                    yield self.batch_2d_boxes[:len(chunks)], self.batch_2d_waypoints[:len(chunks)], None
            if self.endless:
                self.state = None
            else:
                enabled = False

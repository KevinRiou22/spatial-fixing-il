import numpy as np
import torch
import os
import sys
sys.path.append('../model/common')
from config import config as cfg



def fetch_examples():
    examples = []
    for i in range(300):
        examples.append('example{}'.format(i))
    return examples
    

def fetch_simdata_frm_scrw(path, sub_list, debug, data_type="waypoints"):
    keypoints = {}
    if data_type == "waypoints":
        datapath = '../model/data/RLBench_pose_est_data/2d_wp_to_8d_wp_1_frame.npz'
    elif data_type =="images_idxs":
        datapath = '../model/data/RLBench_pose_est_data/2d_wp_to_8d_wp_1_frame_images.npz'

    # if debug:
    #     keypoints_or = {}
    #     for sub in [1, 5, 6, 7, 8, 9, 11]:
    #         data_dbg = '../data/h36m_sub{}.npz'.format(sub)
    #         keypoint_or = np.load(data_dbg, allow_pickle=True)
    #         keypoints_or_metadata = keypoint_or['metadata'].item()
    #         keypoints_or_symmetry = keypoints_or_metadata['keypoints_symmetry']
    #         keypoints_or['S{}'.format(sub)] = keypoint_or['positions_2d'].item()['S{}'.format(sub)]
    for task in sub_list:
        #datapath = path + "/sim_sub{}_300exp.npz".format(sub)
        keypoints['T{}'.format(task)] = {}
    print(datapath)
    assert os.path.isfile(datapath)
    keypoint = np.load(datapath, allow_pickle=True)
    for tk_exp in keypoint['S1'].item().keys():
        task_inx = tk_exp.split('_', 2)
        if 'T{}'.format(task_inx[1]) not in keypoints.keys():
            keypoints['T{}'.format(task_inx[1])] = {}
        keypoints['T{}'.format(task_inx[1])][task_inx[2]] = keypoint['S1'].item()[tk_exp]
    print('Processing done!')
    
    return keypoints


def fetch(keypoints, subjects, action_filter=None,  parse_3d_poses=True, is_test = False):
    out_poses_3d = []
    out_poses_2d_view1 = []
    out_poses_2d_view2 = []
    out_poses_2d_view3 = []
    out_poses_2d_view4 = []
    out_camera_params = []
    out_subject_action = []
    used_cameras = cfg.H36M_DATA.TEST_CAMERAS if is_test else cfg.H36M_DATA.TRAIN_CAMERAS
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a) and len(action.split(a)[1]) <3:
                        found = True
                        break
                if not found:
                    continue
                
            poses_2d = keypoints[subject][action]
            out_subject_action.append([subject, action])
            n_frames = poses_2d[0].shape[0]
            # vis_name_1 = '{}_{}.{}'.format(subject, action, 0)
            # vis_name_2 = '{}_{}.{}'.format(subject, action, 1)
            # vis_name_3 = '{}_{}.{}'.format(subject, action, 2)
            # vis_name_4 = '{}_{}.{}'.format(subject, action, 3)
            # vis_score_cam0 = vis_score[vis_name_1][:n_frames][...,np.newaxis]
            # vis_score_cam1 = vis_score[vis_name_2][:n_frames][...,np.newaxis]
            # vis_score_cam2 = vis_score[vis_name_3][:n_frames][...,np.newaxis]
            # vis_score_cam3 = vis_score[vis_name_4][:n_frames][...,np.newaxis]
            # if vis_score_cam3.shape[0] != vis_score_cam2.shape[0]:
            #     vis_score_cam2 = vis_score_cam2[:-1]
            #     vis_score_cam1 = vis_score_cam1[:-1]
            #     vis_score_cam0 = vis_score_cam0[:-1]
            #     for i in range(4):
            #         poses_2d[i] = poses_2d[i][:-1]
            #
            # if is_test == True and action == 'Walking' and poses_2d[0].shape[0] == 1612:
            #     out_poses_2d_view1.append(np.concatenate((poses_2d[0][1:], vis_score_cam0[1:]), axis =-1))
            #     out_poses_2d_view2.append(np.concatenate((poses_2d[1][1:], vis_score_cam1[1:]), axis =-1))
            #     out_poses_2d_view3.append(np.concatenate((poses_2d[2][1:], vis_score_cam2[1:]), axis =-1))
            #     out_poses_2d_view4.append(np.concatenate((poses_2d[3][1:], vis_score_cam3[1:]), axis =-1))
            # else:
            #     out_poses_2d_view1.append(np.concatenate((poses_2d[0], vis_score_cam0), axis =-1))
            #     out_poses_2d_view2.append(np.concatenate((poses_2d[1], vis_score_cam1), axis =-1))
            #     out_poses_2d_view3.append(np.concatenate((poses_2d[2], vis_score_cam2), axis =-1))
            #     out_poses_2d_view4.append(np.concatenate((poses_2d[3], vis_score_cam3), axis =-1))
            out_poses_2d_view1.append(poses_2d[0])
            out_poses_2d_view2.append(poses_2d[1])
            out_poses_2d_view3.append(poses_2d[2])
            out_poses_2d_view4.append(poses_2d[3])
            print(poses_2d[3].shape)

    
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


def main():
    train_actions = []
    r_src = fetch_simdata_frm_scrw('../model/data/RLBench_pose_est_data/', [0], debug=False, data_type="waypoints")
    np.savez('../model/data/RLBench_pose_est_data/sim_all_formal_waypoints_1_frame.npz', pose=r_src)
    r_src = fetch_simdata_frm_scrw('../model/data/RLBench_pose_est_data/', [0], debug=False, data_type="images_idxs")
    np.savez('../model/data/RLBench_pose_est_data/sim_all_formal_box2d_1_frame.npz', pose=r_src)
    #poses_train_2d, subject_action = fetch(cfg.SIM_DATA.TASKS_TRAIN, train_actions)
     


if __name__ == "__main__":
    main()

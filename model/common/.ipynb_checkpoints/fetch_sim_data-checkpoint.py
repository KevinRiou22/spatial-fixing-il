import numpy as np
import torch
import os
import sys

sys.path.append('repo/pose3d/MTF-Transformer/')


def fetch_simdata_frm_scrw(path, sub_list, debug):
    keypoints = {}
    if debug:
        keypoints_or = {}
        for sub in ['1']:
            data_dbg = 'data/h36m_sub{}.npz'.format(sub)
            keypoint_or = np.load(data_dbg, allow_pickle=True)
            keypoints_or_metadata = keypoint_or['metadata'].item()
            keypoints_or_symmetry = keypoints_or_metadata['keypoints_symmetry']
            keypoints_or['S{}'.format(sub)] = keypoint_or['positions_2d'].item()['S{}'.format(sub)]
    for sub in sub_list:
        datapath = path + "/sim_sub{}_300exp.npz".format(sub)
        assert os.path.isfile(datapath)
        keypoint = np.load(datapath, allow_pickle=True)
        keypoints['S{}'.format(sub)] = keypoint['S{}'.format(sub)].item()
    return keypoints


def fetch_simdata(subjects, action_filter=None, parse_3d_poses=True, is_test=False):
    out_poses_3d = []
    out_poses_2d_view1 = []
    out_poses_2d_view2 = []
    out_poses_2d_view3 = []
    out_poses_2d_view4 = []
    out_camera_params = []
    out_subject_exemp = []


def main():
    r_src = fetch_simdata_frm_scrw('data', ['1'], debug=True)


if __name__ == "__main__":
    main()
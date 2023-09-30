from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import FS95_V1
import numpy as np
import random
import os
import argparse
import sys
import json
import time
from PIL import Image
import cv2
import itertools
from scipy.spatial.transform import Rotation as R
import h5py
from multiprocessing import Pool
import matplotlib.pyplot as plt


def get_all_objects_names(path_dataset, config, mode="train"):
    obs_config = ObservationConfig(image_size=(config["obs_dim"][0], config["obs_dim"][1]), record_gripper_closing=True)
    obs_config.set_all(True)
    # obs_config.set_all_high_dim(False)
    # obs_config.set_all_low_dim(True)
    live_demos = True
    train_tasks = FS95_V1['train']
    test_tasks = FS95_V1['test']
    action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME)
    env = Environment(action_mode, obs_config=obs_config, headless=True)
    env.launch()
    all_names = []
    filenames = []
    if mode == 'train':
        if config['train_task_ids'][0] == "*":
            sim_tasks = np.array(train_tasks).tolist()
        else:
            sim_tasks = np.array(train_tasks)[config["train_task_ids"]].tolist()
    else:
        if config['test_task_ids'][0] == "*":
            sim_tasks = np.array(test_tasks)[config["test_task_ids"]].tolist()
        else:
            sim_tasks = np.array(test_tasks).tolist()
    i=0
    for task_i in sim_tasks:
        try:
            task = env.get_task(task_i)
            task.sample_variation()
            task.reset()
            tsk_names = task._task.get_objects_names()
            print(tsk_names)
            all_names = all_names + tsk_names
            stat_file_id = "/" + str(config["train_task_ids"][i])
            filename = path_dataset + "train" + str(stat_file_id) + "/all_RLBench_objects_names.txt"
            filenames.append(filename)
            with open(filename, 'w') as fp:
                for item in tsk_names:
                    # write each item on a new line
                    fp.write("%s\n" % item)
            i+=1
        except:
            print("Get object names failed for task "+str(i))

    print(all_names)
    filename = path_dataset +  "all_RLBench_objects_names.txt"
    filenames.append(filename)
    with open(filename, 'w') as fp:
        for item in all_names:
            # write each item on a new line
            fp.write("%s\n" % item)
    return filenames

def get_cameras_matrixes(config):
    obs_config = ObservationConfig(image_size=(config["obs_dim"][0], config["obs_dim"][1]), record_gripper_closing=True)
    obs_config.set_all(True)
    # obs_config.set_all_high_dim(False)
    # obs_config.set_all_low_dim(True)
    live_demos = True
    train_tasks = FS95_V1['train']
    test_tasks = FS95_V1['test']
    action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME)
    env = Environment(action_mode, obs_config=obs_config, headless=True)
    env.launch()
    all_names = []
    filenames = []
    sim_tasks = np.array(train_tasks).tolist()

    i = 0
    task_i = sim_tasks[0]
    task = env.get_task(task_i)
    task.sample_variation()
    task.reset()
    cams = ['_cam_over_shoulder_left', '_cam_over_shoulder_right', '_cam_overhead', '_cam_front']

    for cam in cams:
        np.save('cameras_matrixes/'+cam+'_intrinsics.npy', np.array(eval("task._scene."+cam+".get_intrinsic_matrix()")))
        np.save('cameras_matrixes/' + cam + '_extrinsics.npy', np.array(eval("task._scene." + cam + ".get_matrix()")))
        """print(cam)
        print(eval("task._scene."+cam+".get_intrinsic_matrix()"))
        print(eval("task._scene." + cam + ".get_matrix()"))"""
    """self._cam_over_shoulder_left 
    self._cam_over_shoulder_right 
    self._cam_overhead 
    self._cam_wrist 
    self._cam_front """


def remove_duplicate_object_names(obj_names_file):
    for filename in obj_names_file:
        all_names = []
        with open(filename, 'r') as fp:
            for line in fp:
                stripped_line = line.strip()
                no_digits = []
                for i in stripped_line:
                    if not i.isdigit():
                        no_digits.append(i)
                # line_list = stripped_line.split()
                no_digits = ''.join(no_digits)
                if no_digits not in all_names:
                    all_names.append(no_digits)
        with open(filename[:-4]+"_cleaned.txt", 'w') as fp:
            for item in all_names:
                # write each item on a new line
                fp.write("%s\n" % item)
        print(all_names)


def load_objects_names(obj_names_file, old_dict=None):
    objects_names = {}
    with open(obj_names_file, 'r') as fp:
        k = 0
        for line in fp:
            stripped_line = line.strip()
            if old_dict is not None:
                if stripped_line not in old_dict:
                    objects_names[stripped_line] = str(k)
                    k += 1
            else:
                objects_names[stripped_line] = str(k)
                k += 1
    # print(objects_names)
    return objects_names


class Waypoints_demo_generator():
    def __init__(self, dataset_path, config, mode="train", skip_task_on_var_fail=True, max_traj_trials=1):
        self.dataset_path = dataset_path
        self.config = config
        self.mode = mode
        self.skip_task_on_var_fail = skip_task_on_var_fail
        self.max_traj_trials = max_traj_trials
    
    def get_demos(self, task_ids):
        fps = 10
        obs_config = ObservationConfig(image_size=(self.config["obs_dim"][0], self.config["obs_dim"][1]), record_gripper_closing=True)
        cameras = self.config['cams']
        obs_config.set_all(True)
        if len(cameras)==0:
            obs_config.set_all_high_dim(False)
        # obs_self.config.set_all_low_dim(True)
        live_demos = True
        train_tasks = FS95_V1['train']
        test_tasks = FS95_V1['test']
        registered_actions = self.config["registered_actions"]
        object_names = load_objects_names("all_RLBench_objects_names_cleaned.txt")
    
        action_mode = ActionMode(eval("ArmActionMode." + str(self.config['action_mode'])))
        env = Environment(action_mode, obs_config=obs_config, headless=self.config['headless_generation'])
        env.launch()
        #task_idx = start_from
        task_counter = 0

        # cameras = ["left_shoulder", "right_shoulder", "overhead", "front", "wrist"]
    
        if self.mode == 'train':
            sim_tasks = np.array(train_tasks)[task_ids].tolist()
        else:
            sim_tasks = np.array(test_tasks)[task_ids].tolist()
        successfull_tasks = []
        fail_logs = open("fail_logs.txt", 'a')
        for task_i in sim_tasks:
            skip_task = False
            task_idx = task_ids[task_counter]
            dir_task = self.dataset_path + self.mode + "/" + str(task_idx)
            try:
                os.mkdir(dir_task)
                print("Directory ", dir_task, " Created ")
            except FileExistsError:
                print("Directory ", dir_task, " already exists")
            print("task  " + str(task_idx))
            try:
                task = env.get_task(task_i)
            except:
                continue
            data = {}
    
            os.mkdir(dir_task + "/obs")
            os.mkdir(dir_task + "/obs_segmentations")
            os.mkdir(dir_task + "/actions")
            os.mkdir(dir_task + "/obj_poses")
            os.mkdir(dir_task + "/pt_clouds")
            os.mkdir(dir_task + "/traj_keypoints")

            for registered_action in registered_actions:
                os.mkdir(dir_task + "/actions/" + registered_action)
            os.mkdir(dir_task + "/lens")
            os.mkdir(dir_task + "/ee_state")
            os.mkdir(dir_task + "/waypoints")
            if len(cameras)>0:
                    for k in cameras:
                        os.mkdir(dir_task + "/obs/" + str(k))
                        os.mkdir(dir_task + "/obs_segmentations/" + str(k))
                        os.mkdir(dir_task + "/pt_clouds/" + str(k))
            # time.sleep(3)
            variation_numbers = []
            demo_collect_sucess = True
            for ex in range(self.config['n_variation_per_task']):
                if skip_task:
                    break
                for k in cameras:
                    # print(dir_task + "/obs/" + str(k) + "/" + str(ex))
                    os.mkdir(dir_task + "/obs/" + str(k) + "/" + str(ex))
                    os.mkdir(dir_task + "/obs_segmentations/" + str(k) + "/" + str(ex))
                    os.mkdir(dir_task + "/pt_clouds/" + str(k) + "/" + str(ex))
                for registered_action in registered_actions:
                    os.mkdir(dir_task + "/actions/" + registered_action + "/" + str(ex))
                os.mkdir(dir_task + "/ee_state/" + str(ex))
                os.mkdir(dir_task + "/waypoints/" + str(ex))
                os.mkdir(dir_task + "/obj_poses/" + str(ex))
                os.mkdir(dir_task + "/traj_keypoints/" + str(ex))

                task.sample_variation()
                variation_numbers.append(task._variation_number)
    
                for d in range(self.config['n_ex_per_variation']):
                    demo_collect_sucess = False
                    trials = 0
                    while (not demo_collect_sucess) and trials < self.max_traj_trials:
                        try:
                            raw_demos, waypoints_ts = task.get_demos(1, live_demos=live_demos)  # n_demos + current trajectory
                            demo_collect_sucess = True
                        except:
                            trials += 1
                    if not demo_collect_sucess:
                        fail_logs.write("task "+str(task_i)+", task_idx "+str(task_idx)+", var " +str(ex)+", ex "+str(d)+str("\n"))
                        if self.skip_task_on_var_fail:
                            skip_task=True
                            break
                        else:
                            continue
                    print(waypoints_ts)
                    print(raw_demos)
                    waypoints_ts = [0] + waypoints_ts[0]

                    dem_len = len(raw_demos[0])
                    print("dem_len : " + str(dem_len))
                    # dem_len = len(waypoints_ts[d])
                    # for cam in cameras:
                    # data[cam] = np.zeros((dem_len, 4, 128, 128))
                    # data[cam] = np.zeros((dem_len, self.config["obs_dim"][0], self.config["obs_dim"][1], self.config["obs_dim"][2]))
                    data["actions"] = {}
                    for registered_action in registered_actions:
                        data["actions"][registered_action] = np.zeros((dem_len, self.config["action_dim"]))
                    data["ee_state"] = np.zeros((dem_len, 3 + 4 + 1))  # ee position + ee quaternion + ee open/close
                    traj_obj_annot = {}
                    traj_keypoints = {} #for 2D-3D keypoints pose estimation
                    if len(cameras) > 0:
                        for cam in cameras:
                            save_vid_path_rgb = dir_task + "/obs/" + cam + "/" + str(ex) + "/" + str(d) + "_rgb.mp4"
                            save_vid_path_depth = dir_task + "/obs/" + cam + "/" + str(ex) + "/" + str(d) + "_depth.mp4"
                            if self.config['get_segmentations']:
                                save_vid_path_seg = dir_task + "/obs_segmentations/" + cam + "/" + str(ex) + "/" + str(d) + ".mp4"
                                #out_seg = cv2.VideoWriter(save_vid_path_seg, cv2.VideoWriter_fourcc(*'mp4v'), fps, (self.config["obs_dim"][0], self.config["obs_dim"][1]), True)
                            #out_rgb = cv2.VideoWriter(save_vid_path_rgb, cv2.VideoWriter_fourcc(*'mp4v'), fps, (self.config["obs_dim"][0], self.config["obs_dim"][1]), True)
                            #out_depth = cv2.VideoWriter(save_vid_path_depth, cv2.VideoWriter_fourcc(*'mp4v'), fps, (self.config["obs_dim"][0], self.config["obs_dim"][1]), False)
                            for t in range(dem_len):
                                # ts = waypoints_ts[d][t]
                                cam_rgb = cam + "_rgb"
                                cam_depth = cam + "_depth"
                                cam_pt_cl = cam + "_point_cloud"
                                # rgb = np.transpose(np.array(eval('raw_demos[d][t].' + str(cam_rgb))), (2, 0, 1))
                                rgb = np.array(eval('raw_demos[0][t].' + str(cam_rgb))).astype(np.uint8)
                                # depth = np.expand_dims(np.array(eval('raw_demos[d][t].' + str(cam_depth))), 0)
                                depth = (np.array(eval('raw_demos[0][t].' + str(cam_depth))) * 256).astype(np.uint8)
                                pt_cloud = (np.array(eval('raw_demos[0][t].' + str(cam_pt_cl))))
                                np.save(dir_task + "/pt_clouds/" + cam + "/" + str(ex) + "/" + str(d) + "_" + str(t) + ".npy", pt_cloud)
                                # data[cam][t, :, :, :] = np.concatenate([rgb, depth], axis=2)
                                #out_rgb.write(rgb)
                                #out_depth.write(depth)
                                cv2.imwrite(dir_task + "/obs/" + cam + "/" + str(ex) + "/" + str(d) + "_rgb_"+str(t)+".png", rgb)
                                cv2.imwrite(dir_task + "/obs/" + cam + "/" + str(ex) + "/" + str(d) + "_depth_"+str(t)+".png", depth)
                                if self.config['get_segmentations']:
                                    cam_seg = cam + "_mask"
                                    seg = np.array(eval('raw_demos[0][t].' + str(cam_seg))).astype(np.uint8)
                                    #out_seg.write(seg)
                                    cv2.imwrite(dir_task + "/obs_segmentations/" + cam + "/" + str(ex) + "/" + str(d)+"_" +str(t)+ ".png", seg)
                                # print(depth)
                            #out_rgb.release()
                            #out_depth.release()
                            #out_seg.release()
                    for t in range(dem_len):
                        for registered_action in registered_actions:
                            if registered_action == "gripper_pose":
                                data["actions"][registered_action][t, :] = np.concatenate(
                                    [np.array(raw_demos[0][min(t + 1, dem_len - 1)].gripper_pose),
                                     np.array([raw_demos[0][min(t + 1, dem_len - 1)].gripper_open])], axis=-1)
                            else:
                                data["actions"][registered_action][t, :] = np.concatenate(
                                    [np.array(eval("raw_demos[0][t]." + registered_action)),
                                     np.array([raw_demos[0][min(t + 1, dem_len - 1)].gripper_open])], axis=-1)
    
                        data["ee_state"][t, :] = np.concatenate(
                            [np.array(raw_demos[0][t].gripper_pose), np.array([raw_demos[0][t].gripper_open])], axis=-1)
                        traj_obj_annot[str(t)] = raw_demos[0][t].obj_annotations['objects']
                        #print(raw_demos[0][t].obj_annotations['waypoints'])
                        traj_keypoints[str(t)] = raw_demos[0][t].obj_annotations['waypoints']

                        #print(data["ee_state"][t, :])
                        #print(traj_obj_annot[str(t)])
                    """for k in cameras:
                        # save_im_path = dir_task + "/obs/" + k + "/" + str(ex) + "/" + str(d) + ".avi"
                        # np.save(dir_task + "/obs/" + k + "/" + str(ex) + "/" + str(d) + ".npy", np.array(data[k]))
                        # Image.fromarray(np.array(data[k][::, :3])).convert("RGB").save(save_im_path)
                        np.save(dir_task + "/obs/" + k + "/" + str(ex) + "/actual_waypoints_ts_" + str(d) + ".npy",
                                np.array(waypoints_ts))"""
                    np.save(dir_task + "/waypoints/" + str(ex) + "/actual_waypoints_ts_" + str(d) + ".npy", np.array(waypoints_ts))
                    for registered_action in registered_actions:
                        np.save(dir_task + "/actions/" + registered_action + "/" + str(ex) + "/" + str(d) + ".npy",
                                np.array(data["actions"][registered_action]))
                    np.save(dir_task + "/ee_state/" + str(ex) + "/" + str(d) + ".npy", np.array(data["ee_state"]))
                    with open(dir_task + "/obj_poses/" + str(ex) + "/" + str(d) + ".json", 'w') as fp:
                        json.dump(traj_obj_annot, fp)
                    print("saving keypoints : ")
                    with open(dir_task + "/traj_keypoints/" + str(ex) + "/" + str(d) + ".json", 'w') as fp:
                        json.dump(traj_keypoints, fp)
    
                """if not demo_collect_sucess:
                    task_idx += 1
                    continue
                else:
                    successfull_tasks.append(task_i)"""
            np.save(dir_task + "/variation_numbers.npy", np.array(variation_numbers))
            task_counter += 1
        fail_logs.close()
def env_meta_data(config, action_range, robomimic_name=''):
    """"{'env_name': 'Lift', 'type': 1,
     'env_kwargs': {'has_renderer': False, 'has_offscreen_renderer': True, 'ignore_done': True, 'use_object_obs': True,
                    'use_camera_obs': True, 'control_freq': 20,
                    'controller_configs': {'type': 'OSC_POSE', 'input_max': 1, 'input_min': -1,
                                           'output_max': [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                                           'output_min': [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5], 'kp': 150,
                                           'damping': 1, 'impedance_mode': 'fixed', 'kp_limits': [0, 300],
                                           'damping_limits': [0, 10], 'position_limits': None,
                                           'orientation_limits': None, 'uncouple_pos_ori': True, 'control_delta': True,
                                           'interpolation': None, 'ramp_ratio': 0.2}, 'robots': ['Panda'],
                    'camera_depths': False, 'camera_heights': 84, 'camera_widths': 84, 'reward_shaping': False,
                    'camera_names': ['agentview', 'robot0_eye_in_hand'], 'render_gpu_device_id': 0}"""
    meta_data={}
    meta_data["env_name"] = config["name"]

    meta_data["type"] = 3

    env_kwargs = {}
    env_kwargs["task_ids"] = config["train_task_ids"]
    env_kwargs["action_range"] = action_range
    env_kwargs["robomimic_name"] = robomimic_name
    env_kwargs["has_renderer"]=False
    env_kwargs["has_offscreen_renderer"] = False
    env_kwargs["ignore_done"] = True
    env_kwargs["use_object_obs"] = True
    env_kwargs["controller_configs"] = {}
    env_kwargs["use_camera_obs"] = True
    #env_kwargs["control_freq"] = True
    env_kwargs["camera_depths"] = config["get_depth"]
    env_kwargs["use_object_obs"] = True
    env_kwargs["robots"] = ["Panda"]
    env_kwargs["camera_heights"] = config["obs_dim"][0]
    env_kwargs["camera_widths"] = config["obs_dim"][0]
    #env_kwargs["reward_shaping"] = False
    env_kwargs["camera_names"] =  ["front", "wrist"]
    #env_kwargs["render_gpu_device_id"] = 0
    meta_data["env_kwargs"] = env_kwargs
    return meta_data

def filter_gripper_close_states(ee_states, obj_poses, wp_hist):
    previous_ee_state = (ee_states[0]*100).astype(int)
    for ts in range(1, ee_states.shape[0]-1):
        if np.array_equal((ee_states[ts]*100).astype(int)[:-1], previous_ee_state[:-1]):
            print(ee_states[ts])
        """print(ee_states[ts])
        print(ee_states[ts]*1000)
        print((ee_states[ts]*1000).astype(int))"""
        input()
        previous_ee_state = (ee_states[ts]*100).astype(int)

def get_trajectories(path_dataset, config, h5py_dataset, action_range="next_ts", visited_states= 'all', goal=None, keep_gripper_close_states=False):
    #if task!=None:
    object_names_to_id = {}
    n_obj_per_img = 0
    for task in config["train_task_ids"]:
        object_names_to_id = load_objects_names(path_dataset+"train/"+str(task)+"/all_RLBench_objects_names_cleaned.txt", old_dict=object_names_to_id)
        print(object_names_to_id)
        file_obj_poses = open(path_dataset+"train/"+str(task)+"/nb_ob_per_img.json", 'r')
        obj_poses = json.load(file_obj_poses)
        n_obj_per_img = max(obj_poses["nb_ob_per_img_max"], n_obj_per_img)
    # else:
    #     object_names_to_id = load_objects_names("./all_RLBench_objects_names_cleaned.txt")
    #     file_obj_poses = open(config["name"]+"/nb_ob_per_img.json", 'r')


    n_obj_classes = len(object_names_to_id) + 1  # +1 for gripper
    #low_dim = (6 + n_obj_classes)
    if config['use_object_quaternions']:
        low_dim = (6 + 4 + 1)
    else:
        low_dim = (6 + 1) #3D box + quaternion + gripper_open
    n_seen_trajs = 0
    total_samples = 0
    train_ids, valid_ids = [], []

    """print(ee_maxs)
    print(ee_mins)
    input()"""
    for mode in ['train', 'test']:
        if mode == 'train':
            tasks = config["train_task_ids"]
        elif mode == 'test':
            tasks = config["test_task_ids"]
        if len(tasks)>0:
            for tsk in tasks:
                ee_maxs = np.load(path_dataset + "train/"+str(tsk)+"/ees_maxs.npy")
                ee_mins = np.load(path_dataset + "train/"+str(tsk)+"/ees_mins.npy")
                obj_pose_maxs = np.load(path_dataset + "train/" + str(tsk) + "/obj_pose_maxs.npy")
                obj_pose_mins = np.load(path_dataset + "train/" + str(tsk) + "/obj_pose_mins.npy")
                ee_ranges = ee_maxs - ee_mins
                obj_pose_ranges = obj_pose_maxs - obj_pose_mins
                print('task '+str(tsk))
                for var in range(config["n_variation_per_task"]):
                    for ex in range(config["n_ex_per_variation"]):
                        traj_grp = h5py_dataset["data"].create_group("demo_"+str(n_seen_trajs))
                        if mode =="train":
                            train_ids.append("demo_"+str(n_seen_trajs))
                        elif mode=="test":
                            valid_ids.append("demo_" + str(n_seen_trajs))
                        hist_action_path = path_dataset + mode + "/" + str(tsk) + "/actions/gripper_pose/" + str(var) + "/" + str(
                            ex) + ".npy"
                        hist_ee_path = path_dataset + mode + "/" + str(tsk) + "/ee_state/" + str(var) + "/" + str(ex) + ".npy"
                        obj_poses_path = path_dataset + mode + "/" + str(tsk) + "/obj_poses/" + str(var) + "/" + str(ex) + ".json"
                        #actions = np.load(hist_action_path)[:-1, :]

                        ee_states = np.load(hist_ee_path)
                        ee_states = (ee_states - ee_mins) / (ee_ranges)
                        print(ee_states[..., 3:7])
                        #ee_states[..., :3] = ee_states[..., :3]*10
                        #ee_states[..., :3] = (ee_states[..., :3] - ee_mins[..., :3]) / (ee_ranges[..., :3])
                        file_obj_poses = open(obj_poses_path, 'r')
                        obj_poses = json.load(file_obj_poses)
                        file_wp_hist = path_dataset + mode + "/" + str(tsk) + "/waypoints/" + str(var) + "/actual_waypoints_ts_" + str(ex) + ".npy"
                        wp_hist = np.load(file_wp_hist)
                        """if (not keep_gripper_close_states) and visited_states=="all":
                            filter_gripper_close_states(ee_states, obj_poses, wp_hist)"""

                        if visited_states == 'wp_only':
                            seq_len = wp_hist.shape[0]-1
                            traj_grp.attrs['num_samples'] = seq_len
                            total_samples += seq_len
                            actions_data = traj_grp.create_dataset("actions", (seq_len,) + ee_states[0, :].shape, dtype='f')
                        else:
                            seq_len = ee_states[1:, :].shape[0]
                            traj_grp.attrs['num_samples']=seq_len
                            total_samples +=seq_len
                            #actions_data = traj_grp.create_dataset("actions", actions.shape, dtype='f')
                            actions_data = traj_grp.create_dataset("actions", (seq_len,) + ee_states[0, :].shape, dtype='f')
                        if action_range == 'next_ts':
                            actions_data[...] = ee_states[1:, :]
                        elif action_range == 'next_wp':
                            #print('next_wp')
                            #print("seq_len : "+str(seq_len))
                            '''hist_video = path_dataset + mode + "/" + str(tsk) + "/obs/front/" + str(var) + "/" + str(ex) + "_rgb.mp4"
                            cap = cv2.VideoCapture(hist_video)
                            frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
                            fc = 0
                            ret = True'''
                            """if (not training) and (demo_id == -1):
                                frameCount=1"""
                            '''while (fc < frameCount and ret):
                                ret, buf[fc] = cap.read()
                                fc += 1
                            cap.release()'''

                            """print(wp_hist)
                            print(ee_states.shape)"""
                            #print(wp_hist)
                            #print("actions_data : "+str(actions_data))
                            if wp_hist.shape[0]==1:
                                wp_hist = np.concatenate([wp_hist, np.array([seq_len])])
                            if visited_states == 'wp_only':
                                #ee_states[0, :].shape + (len(wp_hist) - 1,)
                                wp_hist[-1] = min(wp_hist[-1], ee_states.shape[0] - 1)
                                #print("wp_hist[1:] : "+str(wp_hist[1:]))
                                #print("ee_states[wp_hist[1:], :] : " + str(ee_states[wp_hist[1:], :]))
                                actions_data[:, :] = ee_states[wp_hist[1:], :]
                                """plt.bar(np.arange(ee_states[wp_hist[:], :].shape[0]), ee_states[wp_hist[:], -1])
                                plt.show()
                                input()"""
                                """for act_dat in actions_data:
                                    print(act_dat)"""
                                """for ts in range(seq_len+1):
                                    print(ee_states.shape)
                                    print(wp_hist)
                                    print(wp_hist[wp_hist>ts])
                                    print(ee_states[wp_hist[ts], :])
                                    min(wp_hist[wp_hist>ts])
                                    input()
                                    wp_hist[-1] = min(wp_hist[-1], ee_states.shape[0] - 1)
                                    actions_data[ts-1, :] = ee_states[wp_hist[ts], :]
                                    # print(ee_states[next_wp_ts, :])
                                    # cv2.imwrite("ts.png", buf[ts])
                                    cv2.imwrite("wp.png", buf[wp_hist[ts]])
                                    input()"""
                            else:
                                for ts in range(ee_states[1:, :].shape[0]):
                                    """print(wp_hist)
                                    print(wp_hist[wp_hist>ts])
                                    print(np.min(wp_hist[wp_hist > ts]))"""
                                    next_wp_ts = np.min(wp_hist[wp_hist>ts])
                                    next_wp_ts = min(next_wp_ts, ee_states.shape[0]-1)
                                    actions_data[ts, :] = ee_states[next_wp_ts, :]
                                    # print(ee_states[next_wp_ts, :])
                                    #cv2.imwrite("ts.png", buf[ts])
                                    #cv2.imwrite("wp.png", buf[next_wp_ts])
                                    #print("next")
                                    #input()



                        #rewards_data = traj_grp.create_dataset("rewards", rewards.shape, dtype='f')
                        #rewards_data = rewards

                        # dones_data = traj_grp.create_dataset("dones", dones.shape, dtype='f')
                        # dones_data = dones

                        obs_grp = traj_grp.create_group("obs")

                        ee_pos_data = obs_grp.create_dataset("robot0_eef_pos", (seq_len, ) + (3,), dtype='f')
                        ee_quat_data = obs_grp.create_dataset("robot0_eef_quat", (seq_len, ) + (4,), dtype='f')
                        grip_data = obs_grp.create_dataset("robot0_gripper_qpos", (seq_len, ) + (1,), dtype='f')

                        if visited_states == 'wp_only':
                            ee_pos_data[...] = ee_states[wp_hist[:-1], :3]
                            ee_quat_data[...] = ee_states[wp_hist[:-1], 3:-1]
                            grip_data[...] = np.expand_dims(ee_states[wp_hist[:-1], -1], -1)
                        else:
                            ee_pos_data[...] = ee_states[:-1, :3]
                            ee_quat_data[...] = ee_states[:-1, 3:-1]
                            grip_data[...] = np.expand_dims(ee_states[:-1, -1], -1)

                        obj_data = obs_grp.create_dataset("object", (seq_len, n_obj_per_img*low_dim), dtype='f')
                        traj_obj_poses = np.zeros((seq_len, n_obj_per_img*low_dim))
                        for t in range(seq_len):
                            n_obj = 0
                            ts = wp_hist[t] if visited_states == 'wp_only' else t
                            concat_obj_infos = np.array([])
                            for obj in obj_poses[str(ts)]:
                                normed_pose = (np.array(obj["bbox"])-obj_pose_mins)/obj_pose_ranges  # 2 * normalize_bbox(np.array(obj["bbox"])) - 1
                                #print('normed_pose : ' + str(normed_pose))
                                if config['use_object_quaternions']:
                                    normed_quat = np.array(obj["quat"])
                                    # print('normed_quat : ' + str(normed_quat))
                                    normed_pose = np.concatenate([normed_pose, normed_quat], axis=-1)
                                #print('concat : ' + str(normed_pose))
                                #concat_obj_infos = np.concatenate([indices_to_one_hot(int(object_names_to_id[obj["name"]]), n_obj_classes).flatten(), normed_pose], axis=-1)
                                concat_obj_infos = np.concatenate([np.array([float(object_names_to_id[obj["name"]])/n_obj_classes]), normed_pose], axis=-1)
                                #print(concat_obj_infos)
                                traj_obj_poses[t, n_obj*low_dim:(n_obj+1)*low_dim]=concat_obj_infos
                                n_obj += 1
                        obj_data[...] = traj_obj_poses

                        next_obs_grp = traj_grp.create_group("next_obs")
                        ee_pos_data = next_obs_grp.create_dataset("robot0_eef_pos", (seq_len, ) + (3,), dtype='f')
                        ee_quat_data = next_obs_grp.create_dataset("robot0_eef_quat", (seq_len, ) + (4,), dtype='f')
                        grip_data = next_obs_grp.create_dataset("robot0_gripper_qpos", (seq_len, ) + (1,), dtype='f')

                        if visited_states == 'wp_only':
                            ee_pos_data[...] = ee_states[wp_hist[1:], :3]
                            ee_quat_data[...] = ee_states[wp_hist[1:], 3:-1]
                            grip_data[...] = np.expand_dims(ee_states[wp_hist[1:], -1], -1)
                        else:
                            ee_pos_data[...] = ee_states[1:, :3]
                            ee_quat_data[...] = ee_states[1:, 3:-1]
                            grip_data[...] = np.expand_dims(ee_states[1:, -1], -1)

                        obj_data = next_obs_grp.create_dataset("object", (seq_len, n_obj_per_img * low_dim), dtype='f')
                        traj_obj_poses = np.zeros((seq_len, n_obj_per_img * low_dim))
                        for t in range(1, seq_len+1):
                            n_obj = 0
                            concat_obj_infos = np.array([])
                            ts = wp_hist[t] if visited_states == 'wp_only' else t
                            for obj in obj_poses[str(ts)]:
                                normed_pose = (np.array(obj["bbox"])-obj_pose_mins)/obj_pose_ranges   # 2 * normalize_bbox(np.array(obj["bbox"])) - 1
                                #print('normed_pose : ' + str(normed_pose))
                                if config['use_object_quaternions']:
                                    normed_quat = np.array(obj["quat"])
                                    # print('normed_quat : ' + str(normed_quat))
                                    normed_pose = np.concatenate([normed_pose, normed_quat], axis=-1)
                                #print('concat : '+str(normed_pose))
                                #concat_obj_infos = np.concatenate([indices_to_one_hot(int(object_names_to_id[obj["name"]]), n_obj_classes).flatten(),normed_pose], axis=-1)
                                concat_obj_infos = np.concatenate([np.array([float(object_names_to_id[obj["name"]])/n_obj_classes]), normed_pose], axis=-1)
                                traj_obj_poses[t-1, n_obj * low_dim:(n_obj + 1) * low_dim] = concat_obj_infos
                                n_obj += 1
                        obj_data[...] = traj_obj_poses
                        n_seen_trajs+=1


                        """for act in traj_grp["actions"]:
                            print(act)
                        for ob in traj_grp["obs"]:
                            print(ob)
                            print(np.array(traj_grp["obs"][ob]))
                        for n_ob in traj_grp["next_obs"]:
                            print(n_ob)
                            print(np.array(traj_grp["next_obs"][n_ob]))
                        input("next")"""
    return h5py_dataset, total_samples, train_ids, valid_ids




def init_worker(visited_states_, tsk, cam, ex, var, colors, mask_colors_to_obj_names, obj_poses, traj_obj_poses, objects_strId_to_intId, mode, cam_transf, ee_states=None, wp_hist=None, keypoints=None):
    global _visited_states_
    global _tsk
    global _cam
    global _ex
    global _var
    global _colors
    global _mask_colors_to_obj_names
    global _obj_poses
    global _traj_obj_poses
    global _objects_strId_to_intId
    global _mode
    global _cam_transf
    global _ee_states
    global _wp_hist
    global _keypoints

    _visited_states_ = visited_states_ 
    _tsk = tsk
    _cam = cam
    _ex = ex
    _var = var
    _colors = colors
    _mask_colors_to_obj_names = mask_colors_to_obj_names
    _obj_poses = obj_poses
    _traj_obj_poses = traj_obj_poses
    _objects_strId_to_intId = objects_strId_to_intId
    _mode = mode
    _cam_transf = cam_transf
    _ee_states = ee_states
    _wp_hist = wp_hist
    _keypoints = keypoints
    
def process_one_ts_data_to_pose_from_image(t):
    global visited_states_
    global tsk
    global cam
    global ex
    global var
    global colors
    global mask_colors_to_obj_names
    global obj_poses
    global traj_obj_poses
    global objects_strId_to_intId
    global mode
    global cam_transf

    # print(t)
    n_obj = 0
    ts = wp_hist[t] if visited_states_ == 'wp_only' else t
    concat_obj_infos = np.array([])
    # 2D gt box
    np_path = path_dataset + mode + "/" + str(tsk) + "/obs_segmentations/" + str(cam) + "/" + str(
        var) + "/" + str(ex) + "_" + str(t) + ".png"
    img_path = path_dataset + mode + "/" + str(tsk) + "/obs/" + str(cam) + "/" + str(
        var) + "/" + str(ex) + "_rgb_" + str(t) + ".png"
    depth_path = path_dataset + mode + "/" + str(tsk) + "/obs/" + str(cam) + "/" + str(
        var) + "/" + str(ex) + "_depth_" + str(t) + ".png"
    numpymask = np.mean(cv2.imread(np_path), axis=-1).astype(np.uint8)
    img = np.mean(cv2.imread(img_path), axis=-1).astype(np.uint8)
    depth = np.mean(cv2.imread(depth_path), axis=-1)
    boxes = {}
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # print((i, j))
            # print(str(numpymask[i][j]))
            if str(numpymask[i][j]) in colors:
                obj_name = mask_colors_to_obj_names[str(numpymask[i][j])]
                if obj_name not in boxes.keys():
                    # print('adding key')
                    boxes[obj_name] = {"min_h": i, "max_h": i, "min_w": j, "max_w": j,
                                       "min_z": {"depth": depth[i, j], "position": [j, i]},
                                       "max_z": {"depth": depth[i, j], "position": [j, i]}}
                else:
                    """print('key already exists')
                    print(boxes[obj_name][min_h])
                    print(boxes[obj_name][max_h])
                    print(boxes[obj_name][min_w])
                    print(boxes[obj_name][max_w])"""
                    if i < boxes[obj_name]["min_h"]:
                        boxes[obj_name]["min_h"] = i
                    if i > boxes[obj_name]["max_h"]:
                        boxes[obj_name]["max_h"] = i
                    if j < boxes[obj_name]["min_w"]:
                        boxes[obj_name]["min_w"] = j
                    if j > boxes[obj_name]["max_w"]:
                        boxes[obj_name]["max_w"] = j
                    if depth[i, j] < boxes[obj_name]["min_z"]["depth"]:
                        boxes[obj_name]["min_z"] = {"depth": depth[i, j], "position": [j, i]}
                    if depth[i, j] > boxes[obj_name]["max_z"]["depth"]:
                        boxes[obj_name]["max_z"] = {"depth": depth[i, j], "position": [j, i]}

    """contours, _ = cv2.findContours(numpymask.copy(), 1, 1)  # not copying here will throw an error
    print(contours[0].shape)
    rect = cv2.minAreaRect(
        contours[0])  # basically you can feed this rect into your classifier
    (x, y), (w, h), a = rect  # a - angle

    box = cv2.boxPoints(rect)
    box = np.int0(box)  # turn into ints
    rect2 = cv2.drawContours(numpymask.copy(), [box], 0, (0, 0, 255), 10)"""
    # print(boxes)
    # boxes = dict(boxes)
    r = 255
    g = 0
    b = 0

    """for i, (k, object_bbx) in enumerate(boxes.items()):
        #print(k)
        #print(object_bbx)
        print("min depth :" +str(object_bbx["min_z"]["depth"]))
        print("max depth :" + str(object_bbx["max_z"]["depth"]))
        start_point = (int(object_bbx["min_w"]), int(object_bbx["min_h"]))

        # Ending coordinate, here (220, 220)
        # represents the bottom right corner of rectangle
        end_point = (int(object_bbx["max_w"]), int(object_bbx["max_h"]))

        # Blue color in BGR
        color = (r, g, b)
        r=r-50
        # g = g+10
        # b=b+30
        # Line thickness of 2 px
        thickness = 1

        # Using cv2.rectangle() method
        # Draw a rectangle with blue line borders of thickness of 2 px
        img = cv2.rectangle(img, start_point, end_point, color, thickness)
        img = cv2.circle(img, object_bbx["min_z"]["position"], radius=2, color=color, thickness=-1)
        img = cv2.circle(img, object_bbx["max_z"]["position"], radius=2, color=color, thickness=-1)

        #traj_obj_poses[t, n_obj * low_dim:(n_obj + 1) * low_dim] = normed_pose



    if t==0:
        plt.imshow(img, cmap='gray')
        plt.figure()
        plt.imshow(numpymask)
        plt.show()"""
    # 3D GT box
    obj_tab = []
    for obj in obj_poses[str(ts)]:
        normed_pose = np.array(
            obj["bbox"])  # - obj_pose_mins) / obj_pose_ranges  # 2 * normalize_bbox(np.array(obj["bbox"])) - 1
        # print('normed_pose : ' + str(normed_pose))
        if config['use_object_quaternions']:
            normed_quat = np.array(obj["quat"])
            # print('normed_quat : ' + str(normed_quat))
            normed_pose = np.concatenate([normed_pose, normed_quat], axis=-1)
        # print('concat : ' + str(normed_pose))
        # concat_obj_infos = np.concatenate([indices_to_one_hot(int(object_names_to_id[obj["name"]]), n_obj_classes).flatten(), normed_pose], axis=-1)
        # concat_obj_infos = np.concatenate([np.array([float(object_names_to_id[obj["name"]]) / n_obj_classes]), normed_pose],axis=-1)
        # print(concat_obj_infos)
        # print(obj["id"])
        if obj["id"] in boxes.keys():
            # print("added")
            # print(obj["id"])
            detect_2d_gt = [boxes[obj["id"]]["min_w"], boxes[obj["id"]]["max_w"], boxes[obj["id"]]["min_h"],
                            boxes[obj["id"]]["max_h"], boxes[obj["id"]]["min_z"]["depth"],
                            boxes[obj["id"]]["max_z"]["depth"]]
            #obj_tab.append(np.concatenate([np.array([objects_strId_to_intId[obj["id"]]]), np.array(detect_2d_gt)], axis=-1))
            traj_obj_poses[t, n_obj] = np.concatenate([np.array([objects_strId_to_intId[obj["id"]]]), np.array(detect_2d_gt)], axis=-1)
            #print("t "+str(t) + " : "+str(traj_obj_poses[t, n_obj]))
            # input()
            n_obj += 1
        # input()
    # print(traj_obj_poses[t])
    # input()
    return (t, traj_obj_poses[t])

def process_one_ts_data_to_pose_from_3D(t):
    global _visited_states_
    global _tsk
    global _cam
    global _ex
    global _var
    global _colors
    global _mask_colors_to_obj_names
    global _obj_poses
    global _traj_obj_poses
    global _objects_strId_to_intId
    global _mode
    global _cam_transf
    # print(t)

    # print(t)



    n_obj = 0
    ts = wp_hist[t] if _visited_states_ == 'wp_only' else t
    concat_obj_infos = np.array([])
    np_path = path_dataset + _mode + "/" + str(_tsk) + "/obs_segmentations/" + str(_cam) + "/" + str(
        _var) + "/" + str(_ex) + "_" + str(t) + ".png"
    numpymask = np.mean(cv2.imread(np_path), axis=-1).astype(np.uint8)
    color_in_segm = np.unique(numpymask)
    visible_objects = [_mask_colors_to_obj_names[c]  for c in color_in_segm.astype('str') if c in _colors]
    obj_tab = []
    for obj in _obj_poses[str(ts)]:
        normed_pose = np.array(
            obj["bbox"])  # - obj_pose_mins) / obj_pose_ranges  # 2 * normalize_bbox(np.array(obj["bbox"])) - 1
        # print('normed_pose : ' + str(normed_pose))
        if config['use_object_quaternions']:
            normed_quat = np.array(obj["quat"])
            # print('normed_quat : ' + str(normed_quat))
            normed_pose = np.concatenate([normed_pose, normed_quat], axis=-1)
        # print('concat : ' + str(normed_pose))
        # concat_obj_infos = np.concatenate([indices_to_one_hot(int(object_names_to_id[obj["name"]]), n_obj_classes).flatten(), normed_pose], axis=-1)
        # concat_obj_infos = np.concatenate([np.array([float(object_names_to_id[obj["name"]]) / n_obj_classes]), normed_pose],axis=-1)
        # print(concat_obj_infos)
        # print(obj["id"])
        obj_center = np.array([(normed_pose[1] + normed_pose[0]) / 2, (normed_pose[3] + normed_pose[2]) / 2,
                               (normed_pose[5] + normed_pose[4]) / 2])
        if obj["id"] in visible_objects:
            box_2d = [] #xmin, xmax, ymin, ymax
            box_3d = []
            for x_i in [0, 1]:
                for y_i in [2, 3]:
                    for z_i in [4, 5]:
                        """print("[x_i, y_i, z_i] : "+str([x_i, y_i, z_i]))
                        print("normed_pose[[x_i, y_i, z_i]] : " + str(normed_pose[[x_i, y_i, z_i]]))
                        input()"""
                        # _pose = np.concatenate([normed_pose[[x_i, y_i, z_i]], [1]])
                        quat = np.array(obj["quat"])
                        quat = np.concatenate([np.array([quat[-1]]), quat[
                                                                     :-1]])  # qw is last from pyrep, but first in "common" representations
                        _pose = qv_mult(quat, normed_pose[[x_i, y_i, z_i]] - obj_center)
                        _pose = _pose + obj_center
                        if len(box_3d)==0:
                            box_3d=[_pose[0], _pose[0], _pose[1], _pose[1], _pose[2], _pose[2]]
                        else:
                            if _pose[0]<box_3d[0]:
                                box_3d[0]=_pose[0]
                            elif _pose[0]>box_3d[1]:
                                box_3d[1]=_pose[0]
                            if _pose[1]<box_3d[2]:
                                box_3d[2]=_pose[1]
                            elif _pose[1]>box_3d[3]:
                                box_3d[3]=_pose[1]
                            if _pose[2]<box_3d[4]:
                                box_3d[4]=_pose[2]
                            elif _pose[2]>box_3d[5]:
                                box_3d[5]=_pose[2]
                        
                        # print(_pose)
                        _pose = np.concatenate([_pose, [1]])
                        # _pose[2]=_pose[2]*0.97
                        pt = np.matmul(_cam_transf[_cam], _pose)
                        pt[:2] = (pt[:2] / pt[2])
                        pt[0]=max(pt[0], 0)
                        pt[1] = max(pt[1], 0)
                        #print("numpymask.shape[0] : "+str(numpymask.shape[0]))
                        #print("numpymask.shape[1] : " + str(numpymask.shape[1]))
                        pt[0] = min(pt[0], numpymask.shape[0])
                        pt[1] = min(pt[1], numpymask.shape[1])
                        if len(box_2d)==0:
                            box_2d=[pt[0], pt[0], pt[1], pt[1]]
                        else:
                            if pt[0]<box_2d[0]:
                                box_2d[0]=pt[0]
                            elif pt[0]>box_2d[1]:
                                box_2d[1]=pt[0]
                            if pt[1]<box_2d[2]:
                                box_2d[2]=pt[1]
                            elif pt[1]>box_2d[3]:
                                box_2d[3]=pt[1]
                        
                        

                        # pt = np.matmul(_extrinsics_mtx, _pose)
                        # pt[2] = pt[2]*0.97
                        # pt = np.matmul(intrinsics_mtx, pt)
                        # print("pt : "+str(pt))
                        color = (255, 0, 0)
                        # if obj["id"]=='usb':
                        #img = cv2.circle(img, (int(pt[0]), int(pt[1])), radius=2, color=color, thickness=-1)
            # print("added")
            # print(obj["id"])
            #detect_2d_gt = [boxes[obj["id"]]["min_w"], boxes[obj["id"]]["max_w"], boxes[obj["id"]]["min_h"],
                            #boxes[obj["id"]]["max_h"], boxes[obj["id"]]["min_z"]["depth"],
                            #boxes[obj["id"]]["max_z"]["depth"]]
            box_2d = np.array(box_2d).astype(int)
            #numpymask = cv2.rectangle(numpymask, (box_2d[0], box_2d[2]), (box_2d[1], box_2d[3]), color, 1)
            #print(t)
            #_traj_obj_poses[t, n_obj] = np.concatenate([np.array([_objects_strId_to_intId[obj["id"]]]), np.array(box_2d), normed_pose], axis=-1)
            #obj_tab.append(np.concatenate([np.array([_objects_strId_to_intId[obj["id"]]]), np.array(box_2d), normed_pose], axis=-1))
            obj_tab.append(np.concatenate([np.array([_objects_strId_to_intId[obj["id"]]]), np.array(box_2d), np.array(box_3d)], axis=-1))
            #print(_traj_obj_poses[t, n_obj])
            """print(box_2d)
            if t==0:
                plt.imshow(numpymask)
                plt.show()"""
            #print("t "+str(t) + " : "+str(traj__obj_poses[t, n_obj]))
            # input()
            n_obj += 1
        # input()
    # print(traj__obj_poses[t])
    # input()
    return (t, obj_tab)


def process_one_ts_data_to_pose_from_3D_full_pose(t):
    global _visited_states_
    global _tsk
    global _cam
    global _ex
    global _var
    global _colors
    global _mask_colors_to_obj_names
    global _obj_poses
    global _traj_obj_poses
    global _objects_strId_to_intId
    global _mode
    global _cam_transf
    # print(t)

    n_obj = 0
    ts = wp_hist[t] if _visited_states_ == 'wp_only' else t
    concat_obj_infos = np.array([])
    np_path = path_dataset + _mode + "/" + str(_tsk) + "/obs_segmentations/" + str(_cam) + "/" + str(
        _var) + "/" + str(_ex) + "_" + str(t) + ".png"
    numpymask = np.mean(cv2.imread(np_path), axis=-1).astype(np.uint8)
    color_in_segm = np.unique(numpymask)
    visible_objects = [_mask_colors_to_obj_names[c] for c in color_in_segm.astype('str') if c in _colors]
    obj_tab = []
    for obj in _obj_poses[str(ts)]:
        normed_pose = np.array(
            obj["bbox"])  # - obj_pose_mins) / obj_pose_ranges  # 2 * normalize_bbox(np.array(obj["bbox"])) - 1
        # print('normed_pose : ' + str(normed_pose))
        if config['use_object_quaternions']:
            normed_quat = np.array(obj["quat"])
            # print('normed_quat : ' + str(normed_quat))
            normed_pose = np.concatenate([normed_pose, normed_quat], axis=-1)
        # print('concat : ' + str(normed_pose))
        # concat_obj_infos = np.concatenate([indices_to_one_hot(int(object_names_to_id[obj["name"]]), n_obj_classes).flatten(), normed_pose], axis=-1)
        # concat_obj_infos = np.concatenate([np.array([float(object_names_to_id[obj["name"]]) / n_obj_classes]), normed_pose],axis=-1)
        # print(concat_obj_infos)
        # print(obj["id"])
        obj_center = np.array([(normed_pose[1] + normed_pose[0]) / 2, (normed_pose[3] + normed_pose[2]) / 2,
                               (normed_pose[5] + normed_pose[4]) / 2])
        obj_dims = np.array([(normed_pose[1] - normed_pose[0]), (normed_pose[3] - normed_pose[2]),
                               (normed_pose[5] - normed_pose[4])])
        quat = np.array(obj["quat"])
        quat = np.concatenate([np.array([quat[-1]]), quat[:-1]])  # qw is last from pyrep, but first in "common" representations

        """print(obj_center)
        print(obj_dims)
        print(quat)
        input()"""
        if obj["id"] in visible_objects:
            box_2d = []  # xmin, xmax, ymin, ymax
            for x_i in [0, 1]:
                for y_i in [2, 3]:
                    for z_i in [4, 5]:
                        """print("[x_i, y_i, z_i] : "+str([x_i, y_i, z_i]))
                        print("normed_pose[[x_i, y_i, z_i]] : " + str(normed_pose[[x_i, y_i, z_i]]))
                        input()"""
                        # _pose = np.concatenate([normed_pose[[x_i, y_i, z_i]], [1]])
                        _pose = qv_mult(quat, normed_pose[[x_i, y_i, z_i]] - obj_center)
                        _pose = _pose + obj_center
                        """if len(box_3d) == 0:
                            box_3d = [_pose[0], _pose[0], _pose[1], _pose[1], _pose[2], _pose[2]]
                        else:
                            if _pose[0] < box_3d[0]:
                                box_3d[0] = _pose[0]
                            elif _pose[0] > box_3d[1]:
                                box_3d[1] = _pose[0]
                            if _pose[1] < box_3d[2]:
                                box_3d[2] = _pose[1]
                            elif _pose[1] > box_3d[3]:
                                box_3d[3] = _pose[1]
                            if _pose[2] < box_3d[4]:
                                box_3d[4] = _pose[2]
                            elif _pose[2] > box_3d[5]:
                                box_3d[5] = _pose[2]"""

                        # print(_pose)
                        _pose = np.concatenate([_pose, [1]])
                        # _pose[2]=_pose[2]*0.97
                        pt = np.matmul(_cam_transf[_cam], _pose)
                        pt[:2] = (pt[:2] / pt[2])
                        pt[0] = max(pt[0], 0)
                        pt[1] = max(pt[1], 0)
                        # print("numpymask.shape[0] : "+str(numpymask.shape[0]))
                        # print("numpymask.shape[1] : " + str(numpymask.shape[1]))
                        pt[0] = min(pt[0], numpymask.shape[0])
                        pt[1] = min(pt[1], numpymask.shape[1])
                        if len(box_2d) == 0:
                            box_2d = [pt[0], pt[0], pt[1], pt[1]]
                        else:
                            if pt[0] < box_2d[0]:
                                box_2d[0] = pt[0]
                            elif pt[0] > box_2d[1]:
                                box_2d[1] = pt[0]
                            if pt[1] < box_2d[2]:
                                box_2d[2] = pt[1]
                            elif pt[1] > box_2d[3]:
                                box_2d[3] = pt[1]

                        # pt = np.matmul(_extrinsics_mtx, _pose)
                        # pt[2] = pt[2]*0.97
                        # pt = np.matmul(intrinsics_mtx, pt)
                        # print("pt : "+str(pt))
                        color = (255, 0, 0)
                        # if obj["id"]=='usb':
                        # img = cv2.circle(img, (int(pt[0]), int(pt[1])), radius=2, color=color, thickness=-1)
            # print("added")
            # print(obj["id"])
            # detect_2d_gt = [boxes[obj["id"]]["min_w"], boxes[obj["id"]]["max_w"], boxes[obj["id"]]["min_h"],
            # boxes[obj["id"]]["max_h"], boxes[obj["id"]]["min_z"]["depth"],
            # boxes[obj["id"]]["max_z"]["depth"]]
            box_2d = np.array(box_2d).astype(int)
            # numpymask = cv2.rectangle(numpymask, (box_2d[0], box_2d[2]), (box_2d[1], box_2d[3]), color, 1)
            # print(t)
            # _traj_obj_poses[t, n_obj] = np.concatenate([np.array([_objects_strId_to_intId[obj["id"]]]), np.array(box_2d), normed_pose], axis=-1)
            # obj_tab.append(np.concatenate([np.array([_objects_strId_to_intId[obj["id"]]]), np.array(box_2d), normed_pose], axis=-1))
            obj_tab.append(np.concatenate([np.array([_objects_strId_to_intId[obj["id"]]]), np.array(box_2d), obj_center, obj_dims, quat], axis=-1))
            # print(_traj_obj_poses[t, n_obj])
            """print(box_2d)
            if t==0:
                plt.imshow(numpymask)
                plt.show()"""
            # print("t "+str(t) + " : "+str(traj__obj_poses[t, n_obj]))
            # input()
            n_obj += 1
        # input()
    # print(traj__obj_poses[t])
    # input()
    return (t, obj_tab)

def process_one_ts_keypoints(t):
    global _visited_states_
    global _tsk
    global _cam
    global _ex
    global _var
    global _colors
    global _mask_colors_to_obj_names
    global _obj_poses
    global _traj_obj_poses
    global _objects_strId_to_intId
    global _mode
    global _cam_transf
    global _wp_hist
    global _ee_poses
    global _keypoints

    n_obj = 0
    #visited_states_ = 'wp_only'
    ts = _keypoints[str(t)] #if visited_states_ == 'wp_only' else t

    concat_obj_infos = np.array([])
    # 2D gt box
    np_path = path_dataset + _mode + "/" + str(_tsk) + "/obs_segmentations/" + str(_cam) + "/" + str(
        _var) + "/" + str(_ex) + "_" + str(t) + ".png"
    img_path = path_dataset + _mode + "/" + str(_tsk) + "/obs/" + str(_cam) + "/" + str(
        _var) + "/" + str(_ex) + "_rgb_" + str(t) + ".png"
    depth_path = path_dataset + _mode + "/" + str(_tsk) + "/obs/" + str(_cam) + "/" + str(
        _var) + "/" + str(_ex) + "_depth_" + str(t) + ".png"
    numpymask = np.mean(cv2.imread(np_path), axis=-1).astype(np.uint8)
    img = np.mean(cv2.imread(img_path), axis=-1).astype(np.uint8)
    depth = np.mean(cv2.imread(depth_path), axis=-1)
    boxes = {}
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # print((i, j))
            # print(str(numpymask[i][j]))
            if str(numpymask[i][j]) in _colors:
                obj_name = _mask_colors_to_obj_names[str(numpymask[i][j])]
                if obj_name not in boxes.keys():
                    # print('adding key')
                    boxes[obj_name] = {"min_h": i, "max_h": i, "min_w": j, "max_w": j,
                                       "min_z": {"depth": depth[i, j], "position": [j, i]},
                                       "max_z": {"depth": depth[i, j], "position": [j, i]}}
                else:
                    """print('key already exists')
                    print(boxes[obj_name][min_h])
                    print(boxes[obj_name][max_h])
                    print(boxes[obj_name][min_w])
                    print(boxes[obj_name][max_w])"""
                    if i < boxes[obj_name]["min_h"]:
                        boxes[obj_name]["min_h"] = i
                    if i > boxes[obj_name]["max_h"]:
                        boxes[obj_name]["max_h"] = i
                    if j < boxes[obj_name]["min_w"]:
                        boxes[obj_name]["min_w"] = j
                    if j > boxes[obj_name]["max_w"]:
                        boxes[obj_name]["max_w"] = j
                    if depth[i, j] < boxes[obj_name]["min_z"]["depth"]:
                        boxes[obj_name]["min_z"] = {"depth": depth[i, j], "position": [j, i]}
                    if depth[i, j] > boxes[obj_name]["max_z"]["depth"]:
                        boxes[obj_name]["max_z"] = {"depth": depth[i, j], "position": [j, i]}

    # print(t)

    n_obj = 0
    #ts = wp_hist[t] if _visited_states_ == 'wp_only' else t
    concat_obj_infos = np.array([])
    #np_path = path_dataset + _mode + "/" + str(_tsk) + "/obs_segmentations/" + str(_cam) + "/" + str(_var) + "/" + str(_ex) + "_" + str(t) + ".png"
    #numpymask = np.mean(cv2.imread(np_path), axis=-1).astype(np.uint8)

    image_path = path_dataset + _mode + "/" + str(_tsk) + "/obs/" + str(_cam) + "/" + str(_var) + "/" + str(_ex) + "_" + str(t) + ".png"
    image = np.mean(cv2.imread(np_path), axis=-1).astype(np.uint8)

    color_in_segm = np.unique(numpymask)
    visible_objects = [_mask_colors_to_obj_names[c] for c in color_in_segm.astype('str') if c in _colors]
    obj_tab = []
    frame_keypoints = _keypoints[str(t)]
    #print("visible objects ex {}, cam {}, t {} : {}".format(_ex, _cam, t, str(visible_objects), ))
    #print(frame_keypoints)
    for i, (kp_id, kp) in enumerate(frame_keypoints.items()):
        #print("parent id objects ex {}, cam {}, t {} : {}".format(_ex, _cam, t, str(kp["parent_id"]), ))

        gt_pos = kp["pos"]
        gt_quat = kp["quat"]
        if "close_gripper()" in kp["ext_string"]:
            gt_grip=np.array([-1.])
        elif "open_gripper()" in kp["ext_string"]:
            gt_grip = np.array([1.])
        else :
            print('kp["ext_string"] : '+str(kp["ext_string"]))
            gt_grip=np.array([0.])
        #gt_grip =  kp["ext_string"]
        #print("gt_pos : "+str(gt_pos))
        #print("gt_quat : " + str(gt_quat))
        #print("gt_grip : " + str(gt_grip))
        gt_3D = np.concatenate([gt_pos, gt_quat, gt_grip])

        """if kp["parent_id"] not in visible_objects:
            uncertainty = 0
            visibility = 0
        else:
            uncertainty = 0
            visibility = 1"""
        uncertainty = np.random.normal(loc=0.0, scale=0.00504, size=(2, ))
        #uncertainty = np.random.normal(loc=0.0, scale=0.1, size=(2,))
        visibility = np.mean(1-uncertainty)
        obj_id = kp["parent_id"]
        print("parent id : "+str(obj_id))
        #print("parent name : {}, parent id : {}".format(kp["parent_id"], obj_id))
        # pt_2d = np.matmul(_cam_transf[_cam], np.concatenate([np.array(gt_3D), [1]]))
        # pt_2d[:2] = (pt_2d[:2] / pt_2d[2])
        # 
        # pt_2d_gt=pt_2d[:2].copy()
        # pt_2d_gt[0] = max(pt_2d_gt[0], 0)
        # pt_2d_gt[1] = max(pt_2d_gt[1], 0)
        # pt_2d_gt[0] = min(pt_2d_gt[0], numpymask.shape[0])
        # pt_2d_gt[1] = min(pt_2d_gt[1], numpymask.shape[1])
        # 
        # pt_2d_pred = pt_2d[:2].copy()
        # pt_2d_pred[0] = pt_2d_pred[0] + uncertainty[0] * numpymask.shape[0]
        # pt_2d_pred[1] = pt_2d_pred[1] + uncertainty[1] * numpymask.shape[1]
        # pt_2d_pred[0] = max(pt_2d_pred[0], 0)
        # pt_2d_pred[1] = max(pt_2d_pred[1], 0)
        # pt_2d_pred[0] = min(pt_2d_pred[0], numpymask.shape[0])
        # pt_2d_pred[1] = min(pt_2d_pred[1], numpymask.shape[1])
        # 
        # #inp_2D = pt_2d[:2]#+uncertainty*256
        # image = cv2.circle(image, (int(pt_2d_gt[0]), int(pt_2d_gt[1])), radius=2, color = (255, 255, 255), thickness=-1)
        # image = cv2.circle(image, (int(pt_2d_pred[0]), int(pt_2d_pred[1])), radius=2, color=(0, 0, 0), thickness=-1)
        #obj_tab.append(np.concatenate( [obj_id, inp_2D, gt_3D, np.array([visibility])], axis=-1))
        #print(boxes.keys())

        #print(_obj_poses[str(t)])
        if obj_id in _objects_strId_to_intId.keys() and obj_id not in boxes:
            print(str(obj_id) + " not in visible objects!")
            detect_2d_gt=[0,0,0,0]
        else:
            if obj_id not in boxes:

                print(str(obj_id) + " not in known segmentable objects!")

                for o in _obj_poses[str(t)]:
                    if o['name']==obj_id:
                        box_3d =np.array(o['bbox'])
                        obj_center = np.array([(box_3d[1] + box_3d[0]) / 2, (box_3d[3] + box_3d[2]) / 2,
                                               (box_3d[5] + box_3d[4]) / 2])
                        quat=np.array(o['quat'])
                        box_2d = []  # xmin, xmax, ymin, ymax
                        print(str(obj_id) + " found in scene objects")
                        for x_i in [0, 1]:
                            for y_i in [2, 3]:
                                for z_i in [4, 5]:
                                    """print("[x_i, y_i, z_i] : "+str([x_i, y_i, z_i]))
                                    print("normed_pose[[x_i, y_i, z_i]] : " + str(normed_pose[[x_i, y_i, z_i]]))
                                    input()"""
                                    # _pose = np.concatenate([normed_pose[[x_i, y_i, z_i]], [1]])
                                    _pose = qv_mult(quat, box_3d[[x_i, y_i, z_i]] - obj_center)
                                    _pose = _pose + obj_center

                                    # print(_pose)
                                    _pose = np.concatenate([_pose, [1]])
                                    # _pose[2]=_pose[2]*0.97
                                    pt = np.matmul(_cam_transf[_cam], _pose)
                                    pt[:2] = (pt[:2] / pt[2])
                                    pt[0] = max(pt[0], 0)
                                    pt[1] = max(pt[1], 0)
                                    # print("numpymask.shape[0] : "+str(numpymask.shape[0]))
                                    # print("numpymask.shape[1] : " + str(numpymask.shape[1]))
                                    pt[0] = min(pt[0], numpymask.shape[0])
                                    pt[1] = min(pt[1], numpymask.shape[1])
                                    if len(box_2d) == 0:
                                        box_2d = [pt[0], pt[0], pt[1], pt[1]]
                                    else:
                                        if pt[0] < box_2d[0]:
                                            box_2d[0] = pt[0]
                                        elif pt[0] > box_2d[1]:
                                            box_2d[1] = pt[0]
                                        if pt[1] < box_2d[2]:
                                            box_2d[2] = pt[1]
                                        elif pt[1] > box_2d[3]:
                                            box_2d[3] = pt[1]
                        detect_2d_gt=box_2d
                        break
            else:
                detect_2d_gt = [boxes[obj_id]["min_w"], boxes[obj_id]["max_w"], boxes[obj_id]["min_h"],
                                boxes[obj_id]["max_h"]]
        print("detect_2d_gt : "+str(detect_2d_gt))
        print("gt_3D : "+str(gt_3D))

        obj_tab.append(np.concatenate([detect_2d_gt, detect_2d_gt, gt_3D, np.array([visibility])], axis=-1))
    cv2.imwrite("/media/facebooksurrounduser/data/deb/ex{}_cam{}_t{}.jpg".format(_ex, _cam, t), image)
    return (t, obj_tab)

def export_dataset_to_pose_estimation_data(path_dataset, config, use_3D_gt=True, action_range="next_ts", visited_states= 'all', goal=None, keep_gripper_close_states=False):
    visited_states_ = visited_states
    # if task!=None:
    object_names_to_id = {}
    n_obj_per_img = 0
    for task in config["train_task_ids"]:
        object_names_to_id = load_objects_names(
            path_dataset + "train/" + str(task) + "/all_RLBench_objects_names_cleaned.txt", old_dict=object_names_to_id)
        print(object_names_to_id)
        file_obj_poses = open(path_dataset + "train/" + str(task) + "/nb_ob_per_img.json", 'r')
        obj_poses = json.load(file_obj_poses)
        #n_obj_per_img = max(obj_poses["nb_ob_per_img_max"], n_obj_per_img)
    # else:
    #     object_names_to_id = load_objects_names("./all_RLBench_objects_names_cleaned.txt")
    #     file_obj_poses = open(config["name"]+"/nb_ob_per_img.json", 'r')

    #n_obj_classes = len(object_names_to_id) + 1  # +1 for gripper
    # low_dim = (6 + n_obj_classes)

    low_dim = 4+4+3
    n_seen_trajs = 0
    total_samples = 0
    train_ids, valid_ids = [], []
    keypoints = {'S1':{}}
    """print(ee_maxs)
    print(ee_mins)
    input()"""
    cam_transf = {}
    for cam in ["left_shoulder", "right_shoulder", "overhead", "front"]:
        if len(cam.split("_")) == 2:
            cam_conf = "over_" + cam.split("_")[1] + '_' + cam.split("_")[0]
        else:
            cam_conf = cam
        extrinsics_mtx = np.load("./cameras_matrixes/_cam_" + str(cam_conf) + "_extrinsics.npy")
        intrinsics_mtx = np.load("./cameras_matrixes/_cam_" + str(cam_conf) + "_intrinsics.npy")
        # intrinsics_mtx = np.concatenate([intrinsics_mtx, np.zeros((3, 1))], axis=-1)
        # transf = np.matmul(intrinsics_mtx, extrinsics_mtx)
        C = np.expand_dims(extrinsics_mtx[:3, 3], 0).T
        R = extrinsics_mtx[:3, :3]
        R_inv = R.T  # inverse of rot matrix is transpose
        R_inv_C = np.matmul(R_inv, C)
        extrinsics_mtx = np.concatenate((R_inv, -R_inv_C), -1)
        transf = np.matmul(intrinsics_mtx, extrinsics_mtx)
        cam_transf[cam]=np.concatenate([transf, [np.array([0, 0, 0, 1])]])

    if use_3D_gt:
        out_filename_base = "keypoints_from_3D"
    else:
        out_filename_base = "keypoints_from_segmentation"

    if config['use_object_quaternions']:
        out_filename_base = out_filename_base+'_full_pose'


    for mode in ['train']:
        if mode == 'train':
            tasks = config["train_task_ids"]
        elif mode == 'test':
            tasks = config["test_task_ids"]
        if len(tasks) > 0:
            max_nb_obj_per_img = 0
            objects_strId_to_intId = {}
            obj_count = 0
            for tsk in tasks:
                mask_colors_to_obj_names = json.load(open('./obj_masks_colors/' + str(tsk) + '.json', 'r'))
                instances = np.array(list(mask_colors_to_obj_names.values()))
                print(instances)

                if len(np.unique(instances)) > max_nb_obj_per_img:
                    max_nb_obj_per_img = len(np.unique(instances))
                for i, (k, v) in enumerate(mask_colors_to_obj_names.items()):
                    if v not in objects_strId_to_intId.keys():
                        objects_strId_to_intId[v]=obj_count
                        obj_count += 1
            print(max_nb_obj_per_img)
            print(objects_strId_to_intId)
            for tsk in tasks:
                ee_maxs = np.load(path_dataset + "train/" + str(tsk) + "/ees_maxs.npy")
                ee_mins = np.load(path_dataset + "train/" + str(tsk) + "/ees_mins.npy")
                obj_pose_maxs = np.load(path_dataset + "train/" + str(tsk) + "/obj_pose_maxs.npy")
                obj_pose_mins = np.load(path_dataset + "train/" + str(tsk) + "/obj_pose_mins.npy")
                ee_ranges = ee_maxs - ee_mins
                obj_pose_ranges = obj_pose_maxs - obj_pose_mins
                mask_colors_to_obj_names = json.load(open('./obj_masks_colors/'+str(tsk)+'.json', 'r'))
                colors = mask_colors_to_obj_names.keys()
                n_obj_per_img = len(colors)
                print('task ' + str(tsk))
                for var in range(config["n_variation_per_task"]):
                    print('var ' + str(var))
                    for ex in range(config["n_ex_per_variation"]):
                        print('ex ' + str(ex))
                        keypoints['S1']['task_'+str(tsk)+'_exemple_'+str(ex)]=[]

                        #traj_grp = h5py_dataset["data"].create_group("demo_" + str(n_seen_trajs))
                        if mode == "train":
                            train_ids.append("demo_" + str(n_seen_trajs))
                        elif mode == "test":
                            valid_ids.append("demo_" + str(n_seen_trajs))
                        hist_action_path = path_dataset + mode + "/" + str(tsk) + "/actions/gripper_pose/" + str(
                            var) + "/" + str(
                            ex) + ".npy"
                        hist_ee_path = path_dataset + mode + "/" + str(tsk) + "/ee_state/" + str(var) + "/" + str(
                            ex) + ".npy"
                        obj_poses_path = path_dataset + mode + "/" + str(tsk) + "/obj_poses/" + str(var) + "/" + str(
                            ex) + ".json"

                        # actions = np.load(hist_action_path)[:-1, :]

                        ee_states = np.load(hist_ee_path)
                        seq_len = ee_states.shape[0]

                        """ee_states = np.load(hist_ee_path)
                        ee_states = (ee_states - ee_mins) / (ee_ranges)
                        print(ee_states[..., 3:7])"""
                        # ee_states[..., :3] = ee_states[..., :3]*10
                        # ee_states[..., :3] = (ee_states[..., :3] - ee_mins[..., :3]) / (ee_ranges[..., :3])
                        file_obj_poses = open(obj_poses_path, 'r')
                        obj_poses = json.load(file_obj_poses)
                        #print(obj_poses)

                        for cam in ["left_shoulder", "right_shoulder", "overhead", "front"]:
                            #obj_data = obs_grp.create_dataset("object", (seq_len, n_obj_per_img * low_dim), dtype='f')
                            traj_obj_poses = np.zeros((seq_len, max_nb_obj_per_img, low_dim))

                            #for t in range(seq_len):
                            with Pool(initializer=init_worker, initargs=(visited_states_, tsk, cam, ex, var, colors, mask_colors_to_obj_names, obj_poses, traj_obj_poses, objects_strId_to_intId, mode, cam_transf, )) as pool:
                                # call the same function with different data in parallel
                                if use_3D_gt:
                                    if config['use_object_quaternions']:
                                        res = pool.map(process_one_ts_data_to_pose_from_3D_full_pose, range(seq_len))
                                    else:
                                        res = pool.map(process_one_ts_data_to_pose_from_3D, range(seq_len))
                                else:
                                    res = pool.map(process_one_ts_data_to_pose_from_image, range(seq_len))
                            for res_ in res:
                                if len(res_[1])>0:
                                    res_objs = np.array(res_[1])
                                    #print(res_[0])
                                    traj_obj_poses[res_[0], :res_objs.shape[0]] = res_objs
                            #input()
                            #print(traj_obj_poses)
                            keypoints['S1']['task_'+str(tsk)+'_exemple_'+str(ex)].append(traj_obj_poses)
                            #print(keypoints['S1']['task_'+str(tsk)+'_exemple_'+str(ex)])
                            #input()
                        #print(keypoints)
                        n_seen_trajs += 1
                np.savez(out_filename_base+'_task_'+str(tsk)+'.npz', **keypoints)
    np.savez(out_filename_base+'.npz', **keypoints)
    #return h5py_dataset, total_samples, train_ids, valid_ids


def export_dataset_to_keypoints_estimation_data(path_dataset, config, use_3D_gt=True, action_range="next_ts", visited_states= 'all', goal=None, keep_gripper_close_states=False, visualize_data=False):
    visited_states_ = visited_states
    # if task!=None:
    object_names_to_id = {}
    n_obj_per_img = 0
    for task in config["train_task_ids"]:
        object_names_to_id = load_objects_names(
            path_dataset + "train/" + str(task) + "/all_RLBench_objects_names_cleaned.txt", old_dict=object_names_to_id)
        print("object_names_to_id task "+str(task)+" : "+str(object_names_to_id))
        file_obj_poses = open(path_dataset + "train/" + str(task) + "/nb_ob_per_img.json", 'r')
        obj_poses = json.load(file_obj_poses)
        #n_obj_per_img = max(obj_poses["nb_ob_per_img_max"], n_obj_per_img)
    # else:
    #     object_names_to_id = load_objects_names("./all_RLBench_objects_names_cleaned.txt")
    #     file_obj_poses = open(config["name"]+"/nb_ob_per_img.json", 'r')

    #n_obj_classes = len(object_names_to_id) + 1  # +1 for gripper
    # low_dim = (6 + n_obj_classes)
    low_dim =  7  #  2d box + 2D box 'pred' + 3D pose + visibility
    n_seen_trajs = 0
    total_samples = 0
    train_ids, valid_ids = [], []

    """print(ee_maxs)
    print(ee_mins)
    input()"""
    cam_transf = {}
    for cam in ["left_shoulder", "right_shoulder", "overhead", "front"]:
        if len(cam.split("_")) == 2:
            cam_conf = "over_" + cam.split("_")[1] + '_' + cam.split("_")[0]
        else:
            cam_conf = cam
        extrinsics_mtx = np.load("./cameras_matrixes/_cam_" + str(cam_conf) + "_extrinsics.npy")
        intrinsics_mtx = np.load("./cameras_matrixes/_cam_" + str(cam_conf) + "_intrinsics.npy")
        # intrinsics_mtx = np.concatenate([intrinsics_mtx, np.zeros((3, 1))], axis=-1)
        # transf = np.matmul(intrinsics_mtx, extrinsics_mtx)
        C = np.expand_dims(extrinsics_mtx[:3, 3], 0).T
        R = extrinsics_mtx[:3, :3]
        R_inv = R.T  # inverse of rot matrix is transpose
        R_inv_C = np.matmul(R_inv, C)
        extrinsics_mtx = np.concatenate((R_inv, -R_inv_C), -1)
        transf = np.matmul(intrinsics_mtx, extrinsics_mtx)
        cam_transf[cam]=np.concatenate([transf, [np.array([0, 0, 0, 1])]])

    dataset = {'S1': {}}
    dataset_waypoints = {'S1': {}}

    out_filename_base = '_image'
    for mode in ['train']:
        if mode == 'train':
            tasks = config["train_task_ids"]
        elif mode == 'test':
            tasks = config["test_task_ids"]
        if len(tasks) > 0:
            max_nb_obj_per_img = 0
            objects_strId_to_intId = {}
            obj_count = 0
            for tsk in tasks:
                mask_colors_to_obj_names = json.load(open('./obj_masks_colors/' + str(tsk) + '.json', 'r'))
                instances = np.array(list(mask_colors_to_obj_names.values()))
                #print(instances)

                if len(np.unique(instances)) > max_nb_obj_per_img:
                    max_nb_obj_per_img = len(np.unique(instances))
                for i, (k, v) in enumerate(mask_colors_to_obj_names.items()):
                    if v not in objects_strId_to_intId.keys():
                        objects_strId_to_intId[v]=obj_count
                        obj_count += 1
            print(max_nb_obj_per_img)
            print("objects_strId_to_intId : "+str(objects_strId_to_intId))
            for tsk in tasks:
                ee_maxs = np.load(path_dataset + "train/" + str(tsk) + "/ees_maxs.npy")
                ee_mins = np.load(path_dataset + "train/" + str(tsk) + "/ees_mins.npy")
                obj_pose_maxs = np.load(path_dataset + "train/" + str(tsk) + "/obj_pose_maxs.npy")
                obj_pose_mins = np.load(path_dataset + "train/" + str(tsk) + "/obj_pose_mins.npy")
                ee_ranges = ee_maxs - ee_mins
                obj_pose_ranges = obj_pose_maxs - obj_pose_mins
                mask_colors_to_obj_names = json.load(open('./obj_masks_colors/'+str(tsk)+'.json', 'r'))
                colors = mask_colors_to_obj_names.keys()
                n_obj_per_img = len(colors)
                print('task ' + str(tsk))

                for var in range(config["n_variation_per_task"]):
                    print('var ' + str(var))
                    for ex in range(config["n_ex_per_variation"]):
                        print('ex ' + str(ex))
                        dataset['S1']['task_'+str(tsk)+'_exemple_'+str(ex)]=[]
                        dataset_waypoints['S1']['task_'+str(tsk)+'_exemple_'+str(ex)]=[]
                        #traj_grp = h5py_dataset["data"].create_group("demo_" + str(n_seen_trajs))
                        if mode == "train":
                            train_ids.append("demo_" + str(n_seen_trajs))
                        elif mode == "test":
                            valid_ids.append("demo_" + str(n_seen_trajs))
                        hist_action_path = path_dataset + mode + "/" + str(tsk) + "/actions/gripper_pose/" + str(
                            var) + "/" + str(
                            ex) + ".npy"
                        hist_ee_path = path_dataset + mode + "/" + str(tsk) + "/ee_state/" + str(var) + "/" + str(
                            ex) + ".npy"
                        obj_poses_path = path_dataset + mode + "/" + str(tsk) + "/obj_poses/" + str(var) + "/" + str(
                            ex) + ".json"
                        #keypoints_path = path_dataset + mode + "/" + str(tsk) + "/traj_keypoints/" + str(var) + "/" + str(ex) + ".json"
                        #file_keypoints = open(keypoints_path, 'r')
                        #keypoints = json.load(file_keypoints)
                        file_wp_hist = path_dataset + mode + "/" + str(tsk) + "/waypoints/" + str( var) + "/actual_waypoints_ts_" + str(ex) + ".npy"
                        wp_hist = np.load(file_wp_hist)
                        # actions = np.load(hist_action_path)[:-1, :]

                        ee_states = np.load(hist_ee_path)
                        seq_len = ee_states.shape[0]

                        """ee_states = np.load(hist_ee_path)
                        ee_states = (ee_states - ee_mins) / (ee_ranges)
                        print(ee_states[..., 3:7])"""
                        # ee_states[..., :3] = ee_states[..., :3]*10
                        # ee_states[..., :3] = (ee_states[..., :3] - ee_mins[..., :3]) / (ee_ranges[..., :3])
                        file_obj_poses = open(obj_poses_path, 'r')
                        obj_poses = json.load(file_obj_poses)
                        #print(obj_poses)
                        if wp_hist.shape[0] == 1:
                            wp_hist = np.concatenate([wp_hist, np.array([seq_len])])

                        # ee_states[0, :].shape + (len(wp_hist) - 1,)
                        wp_hist[-1] = min(wp_hist[-1], ee_states.shape[0] - 1)
                        # print("wp_hist[1:] : "+str(wp_hist[1:]))
                        # print("ee_states[wp_hist[1:], :] : " + str(ee_states[wp_hist[1:], :]))
                        waypoints = ee_states[wp_hist[1:], :8]
                        for cam in ["left_shoulder", "right_shoulder", "overhead", "front"]:
                            #obj_data = obs_grp.create_dataset("object", (seq_len, n_obj_per_img * low_dim), dtype='f')
                            #traj_obj_poses = np.zeros((seq_len, wp_hist.shape[0], low_dim))
                            #traj_obj_poses = np.zeros((1, wp_hist.shape[0], low_dim))
                            traj_obj_poses = np.zeros((seq_len, 4)) #tsk, var, ex, ts


                            #for t in range(seq_len):
                            # with Pool(initializer=init_worker, initargs=(visited_states_, tsk, cam, ex, var, colors, mask_colors_to_obj_names, obj_poses, traj_obj_poses, objects_strId_to_intId, mode, cam_transf, ee_states, wp_hist, keypoints,)) as pool:
                            #     # call the same function with different data in parallel
                            #     #res = pool.map(process_one_ts_keypoints, range(seq_len))
                            #     res_2d_objs = pool.map(process_one_ts_data_to_pose_from_image, range(seq_len))
                            # for res_ in res_2d_objs:
                            #     if len(res_[1])>0:
                            #         res_objs = np.array(res_[1])
                            #         #print(res_[0])
                            #         #print("res_[0] : "+str(res_[0]))
                            #         #print("res_objs.shape[0] : " + str(res_objs.shape[0]))
                            #         traj_obj_poses[res_[0], :res_objs.shape[0]] = res_objs
                            # for t in range(seq_len):
                            #     n_obj = 0
                            #     ts = wp_hist[t] if visited_states_ == 'wp_only' else t
                            #     concat_obj_infos = np.array([])
                            #     # 2D gt box
                            #     np_path = path_dataset + mode + "/" + str(tsk) + "/obs_segmentations/" + str(
                            #         cam) + "/" + str(
                            #         var) + "/" + str(ex) + "_" + str(t) + ".png"
                            #     img_path = path_dataset + mode + "/" + str(tsk) + "/obs/" + str(cam) + "/" + str(
                            #         var) + "/" + str(ex) + "_rgb_" + str(t) + ".png"
                            #     depth_path = path_dataset + mode + "/" + str(tsk) + "/obs/" + str(cam) + "/" + str(
                            #         var) + "/" + str(ex) + "_depth_" + str(t) + ".png"
                            #     numpymask = np.mean(cv2.imread(np_path), axis=-1).astype(np.uint8)
                            for t in range(seq_len):
                                traj_obj_poses[t]=np.array([tsk, var, ex, t])


                            #print(waypoints)
                            dataset_waypoints['S1']['task_' + str(tsk) + '_exemple_' + str(ex)].append(np.expand_dims(waypoints, axis=0))
                            dataset['S1']['task_'+str(tsk)+'_exemple_'+str(ex)].append(traj_obj_poses)
                            #print(keypoints['S1']['task_'+str(tsk)+'_exemple_'+str(ex)])
                            #input()
                        #print(keypoints)
                        n_seen_trajs += 1
                np.savez(path_dataset+out_filename_base+'_box2d_task_'+str(tsk)+'.npz', **dataset)
                np.savez(path_dataset + out_filename_base + '_waypoints_task_' + str(tsk) + '.npz', **dataset_waypoints)
    np.savez(path_dataset+out_filename_base+'_waypoints.npz', **dataset_waypoints)
    np.savez(path_dataset + out_filename_base + '_box2d.npz', **dataset)
    #return h5py_dataset, total_samples, train_ids, valid_ids

def q_conjugate(q):
    q[1:]= -q[1:]
    return q

def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.array([w, x, y, z])

def qv_mult(q1, v1):
    q2 = np.concatenate([np.array([0.0]), v1])
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]

def visualize_mask_colors(path_dataset, config, tasks=None, action_range="next_ts", visited_states= 'all', goal=None, keep_gripper_close_states=False):
    object_names_to_id = {}
    n_obj_per_img = 0
    for task in config["train_task_ids"]:
        object_names_to_id = load_objects_names(
            path_dataset + "train/" + str(task) + "/all_RLBench_objects_names_cleaned.txt", old_dict=object_names_to_id)
        print(object_names_to_id)
        file_obj_poses = open(path_dataset + "train/" + str(task) + "/nb_ob_per_img.json", 'r')
        obj_poses = json.load(file_obj_poses)
        # n_obj_per_img = max(obj_poses["nb_ob_per_img_max"], n_obj_per_img)
    # else:
    #     object_names_to_id = load_objects_names("./all_RLBench_objects_names_cleaned.txt")
    #     file_obj_poses = open(config["name"]+"/nb_ob_per_img.json", 'r')

    # n_obj_classes = len(object_names_to_id) + 1  # +1 for gripper
    # low_dim = (6 + n_obj_classes)
    if config['use_object_quaternions']:
        low_dim = (2 * (6 + 4))
    else:
        low_dim = (2 * 6)
    n_seen_trajs = 0
    total_samples = 0
    train_ids, valid_ids = [], []
    keypoints = {'S1': {}}
    """print(ee_maxs)
    print(ee_mins)
    input()"""
    for mode in ['train']:
        if tasks==None:
            tasks = config["train_task_ids"]
        if len(tasks) > 0:
            for tsk in tasks:
                #mask_colors_to_obj_names = json.load(open('./obj_masks_colors/' + str(tsk) + '.json', 'r'))
                #colors = mask_colors_to_obj_names.keys()
                #n_obj_per_img = len(colors)
                print('task ' + str(tsk))
                for var in range(config["n_variation_per_task"]):
                    for ex in range(config["n_ex_per_variation"]):
                        keypoints['S1'][str(tsk) + '_' + str(ex)] = []

                        # traj_grp = h5py_dataset["data"].create_group("demo_" + str(n_seen_trajs))
                        if mode == "train":
                            train_ids.append("demo_" + str(n_seen_trajs))
                        elif mode == "test":
                            valid_ids.append("demo_" + str(n_seen_trajs))
                        hist_action_path = path_dataset + mode + "/" + str(tsk) + "/actions/gripper_pose/" + str(
                            var) + "/" + str(
                            ex) + ".npy"
                        hist_ee_path = path_dataset + mode + "/" + str(tsk) + "/ee_state/" + str(var) + "/" + str(
                            ex) + ".npy"
                        obj_poses_path = path_dataset + mode + "/" + str(tsk) + "/obj_poses/" + str(var) + "/" + str(
                            ex) + ".json"
                        # actions = np.load(hist_action_path)[:-1, :]

                        ee_states = np.load(hist_ee_path)
                        seq_len = ee_states.shape[0]

                        """ee_states = np.load(hist_ee_path)
                        ee_states = (ee_states - ee_mins) / (ee_ranges)
                        print(ee_states[..., 3:7])"""
                        # ee_states[..., :3] = ee_states[..., :3]*10
                        # ee_states[..., :3] = (ee_states[..., :3] - ee_mins[..., :3]) / (ee_ranges[..., :3])
                        file_obj_poses = open(obj_poses_path, 'r')
                        obj_poses = json.load(file_obj_poses)
                        # print(obj_poses)

                        for cam in ["left_shoulder", "right_shoulder", "overhead", "front"]:
                            if len(cam.split("_"))==2:
                                cam_conf = "over_"+cam.split("_")[1]+'_'+cam.split("_")[0]
                            else:
                                cam_conf=cam
                            extrinsics_mtx = np.load("./cameras_matrixes/_cam_"+str(cam_conf)+"_extrinsics.npy")
                            intrinsics_mtx = np.load("./cameras_matrixes/_cam_" + str(cam_conf) + "_intrinsics.npy")
                            #intrinsics_mtx = np.concatenate([intrinsics_mtx, np.zeros((3, 1))], axis=-1)
                            #transf = np.matmul(intrinsics_mtx, extrinsics_mtx)
                            C = np.expand_dims(extrinsics_mtx[:3, 3], 0).T
                            R = extrinsics_mtx[:3, :3]
                            R_inv = R.T  # inverse of rot matrix is transpose
                            R_inv_C = np.matmul(R_inv, C)
                            extrinsics_mtx = np.concatenate((R_inv, -R_inv_C), -1)
                            transf = np.matmul(intrinsics_mtx, extrinsics_mtx)
                            transf = np.concatenate(
                                [transf, [np.array([0, 0, 0, 1])]])
                            print(extrinsics_mtx)
                            print(intrinsics_mtx)
                            #input()
                            # obj_data = obs_grp.create_dataset("object", (seq_len, n_obj_per_img * low_dim), dtype='f')
                            traj_obj_poses = np.zeros((seq_len, n_obj_per_img, low_dim))
                            for t in range(seq_len):
                                n_obj = 0
                                ts = wp_hist[t] if visited_states == 'wp_only' else t
                                concat_obj_infos = np.array([])
                                # 2D gt box
                                np_path = path_dataset + mode + "/" + str(tsk) + "/obs_segmentations/" + str(
                                    cam) + "/" + str(
                                    var) + "/" + str(ex) + "_" + str(t) + ".png"
                                img_path = path_dataset + mode + "/" + str(tsk) + "/obs/" + str(cam) + "/" + str(
                                    var) + "/" + str(ex) + "_rgb_" + str(t) + ".png"
                                depth_path = path_dataset + mode + "/" + str(tsk) + "/obs/" + str(cam) + "/" + str(
                                    var) + "/" + str(ex) + "_depth_" + str(t) + ".png"
                                numpymask = np.mean(cv2.imread(np_path), axis=-1).astype(np.uint8)
                                img = np.mean(cv2.imread(img_path), axis=-1).astype(np.uint8)
                                depth = np.mean(cv2.imread(depth_path), axis=-1)
                                boxes = {}
                                # print(boxes)
                                # boxes = dict(boxes)
                                r = 255
                                g = 0
                                b = 0


                                # 3D GT box
                                for obj in obj_poses[str(ts)]:
                                    #normed_pose = np.array([0.324, 0, 0.750])#np.array(obj["bbox"])# - obj_pose_mins) / obj_pose_ranges  # 2 * normalize_bbox(np.array(obj["bbox"])) - 1
                                    normed_pose = np.array(obj["bbox"])
                                    print("object : "+str(obj["id"]))
                                    # print('normed_pose : ' + str(normed_pose))

                                    if config['use_object_quaternions']:
                                        normed_quat = np.array(obj["quat"])
                                        # print('normed_quat : ' + str(normed_quat))
                                        normed_pose = np.concatenate([normed_pose, normed_quat], axis=-1)
                                    print("normed_pose : "+str(normed_pose))
                                    #print("normed_pose[[0, 2, 4]] : "+str(normed_pose[[0, 2, 4]]))
                                    obj_center = np.array([(normed_pose[1]+normed_pose[0])/2, (normed_pose[3]+normed_pose[2])/2,(normed_pose[5]+normed_pose[4])/2])
                                    for x_i in [0, 1]:
                                        for y_i in [2, 3]:
                                            for z_i in [4, 5]:
                                                """print("[x_i, y_i, z_i] : "+str([x_i, y_i, z_i]))
                                                print("normed_pose[[x_i, y_i, z_i]] : " + str(normed_pose[[x_i, y_i, z_i]]))
                                                input()"""
                                                #_pose = np.concatenate([normed_pose[[x_i, y_i, z_i]], [1]])
                                                quat = np.array(obj["quat"])
                                                quat = np.concatenate([np.array([quat[-1]]), quat[:-1]])  # qw is last from pyrep, but first in "common" representations
                                                _pose = qv_mult(quat, normed_pose[[x_i, y_i, z_i]]-obj_center)
                                                #print(_pose)
                                                _pose = np.concatenate([_pose+obj_center, [1]])
                                                #_pose[2]=_pose[2]*0.97
                                                pt = np.matmul(transf, _pose)
                                                pt[:2] = (pt[:2] / pt[2])
                                                #pt = np.matmul(extrinsics_mtx, _pose)
                                                #pt[2] = pt[2]*0.97
                                                #pt = np.matmul(intrinsics_mtx, pt)
                                                #print("pt : "+str(pt))
                                                color = (255, 0, 0)
                                                #if obj["id"]=='usb':
                                                img = cv2.circle(img, (int(pt[0]), int(pt[1])), radius=2, color=color, thickness=-1)


                                    #normed_pose = np.concatenate([normed_pose, [1]])

                                    #print("np.matmul(extrinsics_mtx, normed_pose) : " + str(np.matmul(extrinsics_mtx, normed_pose)))

                                    #print("np.matmul(transf, normed_pose) : "+str(np.matmul(transf, normed_pose)))
                                    # print('concat : ' + str(normed_pose))
                                    # concat_obj_infos = np.concatenate([indices_to_one_hot(int(object_names_to_id[obj["name"]]), n_obj_classes).flatten(), normed_pose], axis=-1)
                                    #concat_obj_infos = np.concatenate([np.array([float(object_names_to_id[obj["name"]]) / n_obj_classes]), normed_pose],axis=-1)
                                    # print(concat_obj_infos)

                                    """if obj["id"] in boxes.keys():
                                        #print(obj["id"])
                                        detect_2d_gt = [boxes[obj["id"]]["min_w"], boxes[obj["id"]]["max_w"], boxes[obj["id"]]["min_h"], boxes[obj["id"]]["max_h"], boxes[obj["id"]]["min_z"]["depth"], boxes[obj["id"]]["max_z"]["depth"]]
                                        traj_obj_poses[t, n_obj] = np.concatenate([np.array(detect_2d_gt), normed_pose], axis=-1)
                                        n_obj += 1"""
                                if t==0:
                                    plt.figure()
                                    plt.imshow(img, cmap='gray')
                                    plt.figure()
                                    plt.imshow(numpymask)
                                    plt.show()
def get_trajectories_separate_objs(path_dataset, config, h5py_dataset, action_range="next_ts", visited_states= 'all', goal=None):
    #if task!=None:
    object_names_to_id = {}
    n_obj_per_img = 0
    for task in config["train_task_ids"]:
        object_names_to_id = load_objects_names(path_dataset+"train/"+str(task)+"/all_RLBench_objects_names_cleaned.txt", old_dict=object_names_to_id)
        print(object_names_to_id)
        file_obj_poses = open(path_dataset+"train/"+str(task)+"/nb_ob_per_img.json", 'r')
        obj_poses = json.load(file_obj_poses)
        n_obj_per_img = max(obj_poses["nb_ob_per_img_max"], n_obj_per_img)
    # else:
    #     object_names_to_id = load_objects_names("./all_RLBench_objects_names_cleaned.txt")
    #     file_obj_poses = open(config["name"]+"/nb_ob_per_img.json", 'r')


    n_obj_classes = len(object_names_to_id) + 1  # +1 for gripper
    #low_dim = (6 + n_obj_classes)
    if config['use_object_quaternions']:
        low_dim = (6 + 4 + 1)
    else:
        low_dim = (6 + 1) #3D box + quaternion + gripper_open
    n_seen_trajs = 0
    total_samples = 0
    train_ids, valid_ids = [], []
    for mode in ['train', 'test']:
        if mode == 'train':
            tasks = config["train_task_ids"]
        elif mode == 'test':
            tasks = config["test_task_ids"]
        if len(tasks)>0:
            for tsk in tasks:
                print('task '+str(tsk))
                for var in range(config["n_variation_per_task"]):
                    for ex in range(config["n_ex_per_variation"]):
                        traj_grp = h5py_dataset["data"].create_group("demo_"+str(n_seen_trajs))
                        if mode =="train":
                            train_ids.append("demo_"+str(n_seen_trajs))
                        elif mode=="test":
                            valid_ids.append("demo_" + str(n_seen_trajs))
                        hist_action_path = path_dataset + mode + "/" + str(tsk) + "/actions/gripper_pose/" + str(var) + "/" + str(
                            ex) + ".npy"
                        hist_ee_path = path_dataset + mode + "/" + str(tsk) + "/ee_state/" + str(var) + "/" + str(ex) + ".npy"
                        obj_poses_path = path_dataset + mode + "/" + str(tsk) + "/obj_poses/" + str(var) + "/" + str(ex) + ".json"
                        #actions = np.load(hist_action_path)[:-1, :]
                        ee_states = np.load(hist_ee_path)
                        file_obj_poses = open(obj_poses_path, 'r')
                        obj_poses = json.load(file_obj_poses)
                        file_wp_hist = path_dataset + mode + "/" + str(tsk) + "/waypoints/" + str(var) + "/actual_waypoints_ts_" + str(ex) + ".npy"
                        wp_hist = np.load(file_wp_hist)
                        if visited_states == 'wp_only':
                            seq_len = wp_hist.shape[0]-1
                            traj_grp.attrs['num_samples'] = seq_len
                            total_samples += seq_len
                            actions_data = traj_grp.create_dataset("actions", (seq_len,) + ee_states[0, :].shape, dtype='f')
                        else:
                            seq_len = ee_states[1:, :].shape[0]
                            traj_grp.attrs['num_samples']=seq_len
                            total_samples +=seq_len
                            #actions_data = traj_grp.create_dataset("actions", actions.shape, dtype='f')
                            actions_data = traj_grp.create_dataset("actions", (seq_len,) + ee_states[0, :].shape, dtype='f')
                        if action_range == 'next_ts':
                            actions_data[...] = ee_states[1:, :]
                        elif action_range == 'next_wp':
                            #print('next_wp')
                            #print("seq_len : "+str(seq_len))
                            '''hist_video = path_dataset + mode + "/" + str(tsk) + "/obs/front/" + str(var) + "/" + str(ex) + "_rgb.mp4"
                            cap = cv2.VideoCapture(hist_video)
                            frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
                            fc = 0
                            ret = True
                            """if (not training) and (demo_id == -1):
                                frameCount=1"""
                            while (fc < frameCount and ret):
                                ret, buf[fc] = cap.read()
                                fc += 1
                            cap.release()'''

                            """print(wp_hist)
                            print(ee_states.shape)"""
                            #print(wp_hist)
                            #print("actions_data : "+str(actions_data))
                            if wp_hist.shape[0]==1:
                                wp_hist = np.concatenate([wp_hist, np.array([seq_len])])
                            if visited_states == 'wp_only':
                                #ee_states[0, :].shape + (len(wp_hist) - 1,)
                                for ts in range(1, seq_len+1):
                                    #print(wp_hist)
                                    #print(wp_hist[wp_hist>ts])
                                    actions_data[ts-1, :] = ee_states[wp_hist[ts], :]
                                    # print(ee_states[next_wp_ts, :])
                                    # cv2.imwrite("ts.png", buf[ts])
                                    # cv2.imwrite("wp.png", buf[next_wp_ts])
                                    # input()
                            else:
                                for ts in range(ee_states[1:, :].shape[0]):
                                    """print(wp_hist)
                                    print(wp_hist[wp_hist>ts])
                                    print(np.min(wp_hist[wp_hist > ts]))"""
                                    next_wp_ts = np.min(wp_hist[wp_hist>ts])
                                    next_wp_ts = min(next_wp_ts, ee_states.shape[0]-1)
                                    actions_data[ts, :] = ee_states[next_wp_ts, :]
                                    # print(ee_states[next_wp_ts, :])
                                    # cv2.imwrite("ts.png", buf[ts])
                                    # cv2.imwrite("wp.png", buf[next_wp_ts])
                                    #input()



                        #rewards_data = traj_grp.create_dataset("rewards", rewards.shape, dtype='f')
                        #rewards_data = rewards

                        # dones_data = traj_grp.create_dataset("dones", dones.shape, dtype='f')
                        # dones_data = dones

                        obs_grp = traj_grp.create_group("obs")

                        ee_pos_data = obs_grp.create_dataset("robot0_eef_pos", (seq_len, ) + (3,), dtype='f')
                        ee_quat_data = obs_grp.create_dataset("robot0_eef_quat", (seq_len, ) + (4,), dtype='f')
                        grip_data = obs_grp.create_dataset("robot0_gripper_qpos", (seq_len, ) + (1,), dtype='f')

                        if visited_states == 'wp_only':
                            ee_pos_data[...] = ee_states[wp_hist[:-1], :3]
                            ee_quat_data[...] = ee_states[wp_hist[:-1], 3:-1]
                            grip_data[...] = np.expand_dims(ee_states[wp_hist[:-1], -1], -1)
                        else:
                            ee_pos_data[...] = ee_states[:-1, :3]
                            ee_quat_data[...] = ee_states[:-1, 3:-1]
                            grip_data[...] = np.expand_dims(ee_states[:-1, -1], -1)
                        obj_i = 0
                        obj_datas = []
                        for obj in obj_poses[str(0)]:
                            obj_datas.append(obs_grp.create_dataset("object_"+str(obj_i), (seq_len, low_dim), dtype='f'))
                            traj_obj_poses = np.zeros((seq_len, low_dim))
                            for t in range(seq_len):
                                n_obj = 0
                                ts = wp_hist[t] if visited_states == 'wp_only' else t
                                concat_obj_infos = np.array([])
                                normed_pose = np.array(obj["bbox"])  # 2 * normalize_bbox(np.array(obj["bbox"])) - 1
                                #print('normed_pose : ' + str(normed_pose))
                                if config['use_object_quaternions']:
                                    normed_quat = np.array(obj["quat"])
                                    # print('normed_quat : ' + str(normed_quat))
                                    normed_pose = np.concatenate([normed_pose, normed_quat], axis=-1)
                                #print('concat : ' + str(normed_pose))
                                #concat_obj_infos = np.concatenate([indices_to_one_hot(int(object_names_to_id[obj["name"]]), n_obj_classes).flatten(), normed_pose], axis=-1)
                                concat_obj_infos = np.concatenate([np.array([float(object_names_to_id[obj["name"]])/n_obj_classes]), normed_pose], axis=-1)
                                #print(concat_obj_infos)
                                traj_obj_poses[t, :]=concat_obj_infos
                                n_obj += 1
                            obj_datas[obj_i][...] = traj_obj_poses
                            obj_i += 1

                        next_obs_grp = traj_grp.create_group("next_obs")
                        ee_pos_data = next_obs_grp.create_dataset("robot0_eef_pos", (seq_len, ) + (3,), dtype='f')
                        ee_quat_data = next_obs_grp.create_dataset("robot0_eef_quat", (seq_len, ) + (4,), dtype='f')
                        grip_data = next_obs_grp.create_dataset("robot0_gripper_qpos", (seq_len, ) + (1,), dtype='f')

                        if visited_states == 'wp_only':
                            ee_pos_data[...] = ee_states[wp_hist[1:], :3]
                            ee_quat_data[...] = ee_states[wp_hist[1:], 3:-1]
                            grip_data[...] = np.expand_dims(ee_states[wp_hist[1:], -1], -1)
                        else:
                            ee_pos_data[...] = ee_states[1:, :3]
                            ee_quat_data[...] = ee_states[1:, 3:-1]
                            grip_data[...] = np.expand_dims(ee_states[1:, -1], -1)

                        obj_datas_next = []
                        obj_i = 0
                        for obj in obj_poses[str(0)]:
                            obj_datas_next.append(next_obs_grp.create_dataset("object_"+str(obj_i), (seq_len, low_dim), dtype='f'))
                            traj_obj_poses = np.zeros((seq_len, low_dim))
                            for t in range(1, seq_len+1):
                                n_obj = 0
                                concat_obj_infos = np.array([])
                                ts = wp_hist[t] if visited_states == 'wp_only' else t
                                normed_pose = np.array(obj["bbox"])  # 2 * normalize_bbox(np.array(obj["bbox"])) - 1
                                #print('normed_pose : ' + str(normed_pose))
                                if config['use_object_quaternions']:
                                    normed_quat = np.array(obj["quat"])
                                    # print('normed_quat : ' + str(normed_quat))
                                    normed_pose = np.concatenate([normed_pose, normed_quat], axis=-1)
                                #print('concat : '+str(normed_pose))
                                #concat_obj_infos = np.concatenate([indices_to_one_hot(int(object_names_to_id[obj["name"]]), n_obj_classes).flatten(),normed_pose], axis=-1)
                                concat_obj_infos = np.concatenate([np.array([float(object_names_to_id[obj["name"]])/n_obj_classes]), normed_pose], axis=-1)
                                traj_obj_poses[t-1, :] = concat_obj_infos
                                n_obj += 1
                            obj_datas[obj_i][...] = traj_obj_poses
                            obj_i+=1
                        n_seen_trajs+=1


                        """for act in traj_grp["actions"]:
                            print(act)
                        for ob in traj_grp["obs"]:
                            print(ob)
                            print(np.array(traj_grp["obs"][ob]))
                        for n_ob in traj_grp["next_obs"]:
                            print(n_ob)
                            print(np.array(traj_grp["next_obs"][n_ob]))
                        input("next")"""
    return h5py_dataset, total_samples, train_ids, valid_ids

def export_dataset_to_robomimic_format(dataset_path, config, action_range="next_ts", visited_states='all', goal=None, robomimic_path = 'low_dim', robomimic_name = 'low_dim', task=None):
    f = h5py.File(str(robomimic_path)+'.hdf5', 'w')
    grp = f.create_group("data")
    grp.attrs["env_args"] = json.dumps(env_meta_data(config, action_range, robomimic_name))
    f, total_samples, train_ids, valid_ids = get_trajectories(dataset_path, config, f, action_range=action_range, visited_states=visited_states, goal=goal)

    grp.attrs["total"] = total_samples
    #['train', 'valid']
    mask_grp = f.create_group("mask")
    print(train_ids)
    mask_train = mask_grp.create_dataset("train", data= np.string_(train_ids))
    #mask_train[...]=train_ids
    if len(valid_ids)>0:
        mask_valid = mask_grp.create_dataset("valid", data=valid_ids)
        mask_valid[...]=valid_ids
    print(f.keys())

def get_dataset_stats(path_dataset, config):
    all_actions_mins = []
    all_actions_maxs = []
    all_ee_mins = []
    all_ee_maxs = []
    nb_ob_per_img_min = []
    nb_ob_per_img_max = []

    mt_all_actions_mins = []
    mt_all_actions_maxs = []
    mt_all_ee_mins = []
    mt_all_ee_maxs = []
    mt_nb_ob_per_img_min = []
    mt_nb_ob_per_img_max = []

    all_obj_pose_mins = []
    all_obj_pose_maxs = []
    mt_all_obj_pose_mins = []
    mt_all_obj_pose_maxs = []

    tasks_lens={}
    tasks_wp_lens = {}
    train_tasks = FS95_V1['train']
    test_tasks = FS95_V1['test']
    for tsk in config["train_task_ids"]:
        all_tsk_lens = []
        all_tsk_wp_lens = []
        for var in range(config["n_variation_per_task"]):
            for ex in range(config["n_ex_per_variation"]):
                try:
                    hist_action_path = path_dataset + "train/" + str(tsk) + "/actions/gripper_pose/" + str(var) + "/" + str(
                        ex) + ".npy"
                    hist_ee_path = path_dataset + "train/" + str(tsk) + "/ee_state/" + str(var) + "/" + str(ex) + ".npy"
                    wp_path = path_dataset + "train/" + str(tsk) + "/waypoints/" + str(var) + "/actual_waypoints_ts_" + str(ex) + ".npy"
                    obj_poses_path = path_dataset + "train/" + str(tsk) + "/obj_poses/" + str(var) + "/" + str(ex) + ".json"
                    actions = np.load(hist_action_path)
                    waypoints = np.load(wp_path)
                    all_tsk_wp_lens.append(waypoints.shape[0])
                    all_tsk_lens.append(actions.shape[0])
                    ee_states = np.load(hist_ee_path)
                    file_obj_poses = open(obj_poses_path, 'r')
                    obj_poses = json.load(file_obj_poses)
                    n_objs = []
                    traj_obj_poses = []
                    for i, (ts, objs) in enumerate(obj_poses.items()):
                        n_objs.append(len(objs))
                        for obj in objs:
                            traj_obj_poses.append(obj["bbox"])
                    all_obj_pose_mins.append(np.min(np.array(traj_obj_poses), axis=0))
                    all_obj_pose_maxs.append(np.max(np.array(traj_obj_poses), axis=0))
                    mt_all_obj_pose_mins.append(np.min(np.array(traj_obj_poses), axis=0))
                    mt_all_obj_pose_maxs.append(np.max(np.array(traj_obj_poses), axis=0))
                    nb_ob_per_img_min.append(min(n_objs))
                    nb_ob_per_img_max.append(max(n_objs))
                    all_actions_mins.append(np.min(actions, axis=0))
                    all_actions_maxs.append(np.max(actions, axis=0))
                    all_ee_mins.append(np.min(ee_states, axis=0))
                    all_ee_maxs.append(np.max(ee_states, axis=0))
                    mt_nb_ob_per_img_min.append(min(n_objs))
                    mt_nb_ob_per_img_max.append(max(n_objs))
                    mt_all_actions_mins.append(np.min(actions, axis=0))
                    mt_all_actions_maxs.append(np.max(actions, axis=0))
                    mt_all_ee_mins.append(np.min(ee_states, axis=0))
                    mt_all_ee_maxs.append(np.max(ee_states, axis=0))
                except:
                    print("Error with task "+str(tsk)+", var "+str(var)+", ex "+str(ex))
        try:
            nb_ob_per_img_min = np.min(np.array(nb_ob_per_img_min))
            nb_ob_per_img_max = np.max(np.array(nb_ob_per_img_max))
            all_actions_mins = np.min(np.array(all_actions_mins), axis=0)
            all_actions_maxs = np.max(np.array(all_actions_maxs), axis=0)
            all_obj_pose_mins = np.min(np.array(all_obj_pose_mins), axis=0)
            all_obj_pose_maxs = np.max(np.array(all_obj_pose_maxs), axis=0)

            all_ee_mins = np.min(np.array(all_ee_mins), axis=0)
            all_ee_maxs = np.max(np.array(all_ee_maxs), axis=0)
            stat_file_id = "/" + str(tsk)
            np.save(path_dataset + "train"+str(stat_file_id)+"/action_mins.npy", all_actions_mins)
            np.save(path_dataset + "train"+str(stat_file_id)+ "/action_maxs.npy", all_actions_maxs)
            np.save(path_dataset + "train"+str(stat_file_id)+ "/ees_mins.npy", all_ee_mins)
            np.save(path_dataset + "train"+str(stat_file_id)+ "/ees_maxs.npy", all_ee_maxs)
            np.save(path_dataset + "train" + str(stat_file_id) + "/obj_pose_mins.npy", all_obj_pose_mins)
            np.save(path_dataset + "train" + str(stat_file_id) + "/obj_pose_maxs.npy", all_obj_pose_maxs)
            with open(path_dataset + "train"+str(stat_file_id)+ "/nb_ob_per_img.json", 'w') as fp:
                json.dump({"nb_ob_per_img_min": int(nb_ob_per_img_min), "nb_ob_per_img_max": int(nb_ob_per_img_max)}, fp)
        except:
            print("Error while trying to compute/save task "+str(tsk)+" stats.")
        all_obj_pose_mins = []
        all_obj_pose_maxs = []
        all_actions_mins = []
        all_actions_maxs = []
        all_ee_mins = []
        all_ee_maxs = []
        nb_ob_per_img_min = []
        nb_ob_per_img_max = []
        all_tsk_lens = np.array(all_tsk_lens)
        all_tsk_wp_lens = np.array(all_tsk_wp_lens)


        tsk_name = vars(train_tasks[tsk])['__module__'].split('.')[-1]
        print(tsk_name)
        try:
            tasks_lens[str(tsk)] = {'name':tsk_name, 'mean_len' : float(np.mean(all_tsk_lens)), 'max_len' : float(np.max(all_tsk_lens)), 'min_len' : float(np.min(all_tsk_lens)), 'std_len' : float(np.std(all_tsk_lens)), 'n_ex':int(all_tsk_lens.shape[0])}
            tasks_wp_lens[str(tsk)] = {'name':tsk_name, 'mean_len' : float(np.mean(all_tsk_wp_lens)), 'max_len' : float(np.max(all_tsk_wp_lens)), 'min_len' : float(np.min(all_tsk_wp_lens)), 'std_len' : float(np.std(all_tsk_wp_lens)), 'n_ex':int(all_tsk_wp_lens.shape[0])}

        except:
            tasks_lens[str(tsk)] = {'name':tsk_name, 'mean_len' : -1, 'max_len' : -1, 'min_len' : -1, 'std_len' : -1, 'n_ex':0}
            tasks_wp_lens[str(tsk)] = {'name': tsk_name, 'mean_len': -1, 'max_len': -1, 'min_len': -1, 'std_len': -1, 'n_ex': 0}
            print("no tsk lenght data to process for tsk "+str(tsk))
    mt_nb_ob_per_img_min = np.min(np.array(mt_nb_ob_per_img_min))
    mt_nb_ob_per_img_max = np.max(np.array(mt_nb_ob_per_img_max))
    mt_all_actions_mins = np.min(np.array(mt_all_actions_mins), axis=0)
    mt_all_actions_maxs = np.max(np.array(mt_all_actions_maxs), axis=0)
    mt_all_ee_mins = np.min(np.array(mt_all_ee_mins), axis=0)
    mt_all_ee_maxs = np.max(np.array(mt_all_ee_maxs), axis=0)
    mt_all_obj_pose_mins = np.min(np.array(mt_all_obj_pose_mins), axis=0)
    mt_all_obj_pose_maxs = np.max(np.array(mt_all_obj_pose_maxs), axis=0)

    np.save(path_dataset + "action_mins.npy", mt_all_actions_mins)
    np.save(path_dataset + "action_maxs.npy", mt_all_actions_maxs)

    np.save(path_dataset + "ees_mins.npy", mt_all_ee_mins)
    np.save(path_dataset + "ees_maxs.npy", mt_all_ee_maxs)

    np.save(path_dataset + "obj_pose_mins.npy", mt_all_obj_pose_mins)
    np.save(path_dataset + "obj_pose_maxs.npy", mt_all_obj_pose_maxs)

    #print({"nb_ob_per_img_min": int(mt_nb_ob_per_img_min), "nb_ob_per_img_max": int(mt_nb_ob_per_img_max)})
    with open(path_dataset + "nb_ob_per_img.json",'w') as fp:
        json.dump({"nb_ob_per_img_min": int(mt_nb_ob_per_img_min), "nb_ob_per_img_max": int(mt_nb_ob_per_img_max)}, fp)
    #print(tasks_lens)
    with open(path_dataset + "tasks_lenght_stats.json",'w') as fp_len:
        json.dump(tasks_lens, fp_len)

    with open(path_dataset + "tasks_wp_lenght_stats.json",'w') as fp_wp_len:
        json.dump(tasks_wp_lens, fp_wp_len)
def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


class Dataset():
    def __init__(self, dataset_path, batch_size, n_demos, cams, n_local_ts, n_global_ts, ts_ov_wp=2, depth=False):
        file_config = open(dataset_path + 'dataset_config.json', 'r')

        self.config = json.load(file_config)
        self.path_dataset = dataset_path
        self.batch_size = batch_size
        self.n_demos = n_demos
        self.used_cams = cams
        self.n_cameras = len(self.used_cams)
        self.n_local_ts = n_local_ts
        self.n_global_ts = n_global_ts
        self.ts_ov_wp_ratio = ts_ov_wp
        self.get_depth = depth

        self.object_names_to_id = load_objects_names(
            "datasets/" + self.config['env'] + '/' + "all_RLBench_objects_names_cleaned.txt")
        file_obj_poses = open("datasets/" + self.config['env'] + '/' + self.config['name'] + "/nb_ob_per_img.json", 'r')
        self.obj_poses = json.load(file_obj_poses)
        self.n_obj_per_img = self.obj_poses["nb_ob_per_img_max"]
        self.n_obj_classes = len(self.object_names_to_id)
        if use_object_quaternions:
            self.low_dim = (6 + 4 + 1)
        else:
            self.low_dim = (6 + 1)
        self.n_obj = self.n_obj_per_img
        # self.seen_trajetories = np.ones(len(self.config["train_task_ids"]), self.config["n_variation_per_task"], self.config["n_ex_per_variation"]).astype(bool)
        tasks = self.config["train_task_ids"]
        variations = np.arange(self.config["n_variation_per_task"])
        exemples = np.arange(self.config["n_ex_per_variation"])
        self.all_exemples = np.array(list(itertools.product(*[tasks, variations, exemples])))
        self.all_exemples_ids = np.arange(self.all_exemples.shape[0])
        self.train_set_size = self.all_exemples.shape[0]
        self.n_epochs = -1
        self.prepare_epoch()

        self.act_mins = np.load(self.path_dataset + "action_mins.npy")
        self.act_maxs = np.load(self.path_dataset + "action_maxs.npy")
        self.ee_mins = np.load(self.path_dataset + "ees_mins.npy")
        self.ee_maxs = np.load(self.path_dataset + "ees_maxs.npy")
        self.euler_maxs = np.load(self.path_dataset + "euler_maxs.npy")
        self.euler_mins = np.load(self.path_dataset + "euler_mins.npy")
        self.euler_range = self.euler_maxs - self.euler_mins
        self.ee_range = self.ee_maxs - self.ee_mins
        self.action_range = self.act_maxs - self.act_mins

        """n_tot_ts = self.n_global_ts + (self.n_global_ts - 1) * self.ts_ov_wp_ratio
        rep_act_min = np.repeat(np.expand_dims(self.act_mins, 0), self.n_local_ts, axis=0)
        self.rep_act_min = np.repeat(np.expand_dims(rep_act_min, 0), n_tot_ts, axis=0)
        rep_act_max = np.repeat(np.expand_dims(self.act_maxs, 0), self.n_local_ts, axis=0)
        rep_act_max = np.repeat(np.expand_dims(rep_act_max, 0), n_tot_ts, axis=0)
        self.rep_act_range=rep_act_max[::, 0:3] - self.rep_act_min[::, 0:3]

        rep_ee_pos_min = np.repeat(np.expand_dims(self.ee_mins, 0), self.n_local_ts, axis=0)
        self.rep_ee_pos_min = np.repeat(np.expand_dims(rep_ee_pos_min, 0), n_tot_ts, axis=0)
        rep_ee_pos_max = np.repeat(np.expand_dims(self.ee_maxs, 0), self.n_local_ts, axis=0)
        rep_ee_pos_max = np.repeat(np.expand_dims(rep_ee_pos_max, 0), n_tot_ts, axis=0)
        self.rep_ee_pos_range = rep_ee_pos_max - self.rep_ee_pos_min

        rep_euler_min = np.repeat(np.expand_dims(self.euler_mins, 0), self.n_local_ts, axis=0)
        self.rep_euler_min = np.repeat(np.expand_dims(rep_euler_min, 0), n_tot_ts, axis=0)
        rep_euler_max = np.repeat(np.expand_dims(self.euler_maxs, 0), self.n_local_ts, axis=0)
        rep_euler_max = np.repeat(np.expand_dims(rep_euler_max, 0), n_tot_ts, axis=0)
        self.rep_euler_range = rep_euler_max - self.rep_euler_min"""

    def euler_from_quaternion(self, quats):
        if len(quats.shape) > 2:
            eul_rots = [[] for k in range(quats.shape[0])]
            for k in range(quats.shape[0]):
                for i in range(quats.shape[1]):
                    if quats[k][i].all() == np.zeros((4)).all():
                        eul_rots[k].append(np.zeros((3)))
                    else:
                        r = R.from_quat(quats[k][i])
                        eul_rots[k].append(r.as_euler('xyz', degrees=False))

        else:
            eul_rots = []
            for k in range(quats.shape[0]):
                r = R.from_quat(quats[k])
                eul_rots.append(r.as_euler('xyz', degrees=False))
        return np.array(eul_rots)

    def quaternion_from_euler(self, rots):
        if len(rots.shape) > 2:
            quats = [[] for k in range(rots.shape[0])]
            for k in range(rots.shape[0]):
                for i in range(rots.shape[1]):
                    quat_back = R.from_euler('xyz', rots[k][i], degrees=False)
                    quats[i].append(quat_back.as_quat())
        else:
            quats = []
            for k in range(rots.shape[0]):
                quat_back = R.from_euler('xyz', rots[k], degrees=False)
                quats.append(quat_back.as_quat())
        return -1 * np.array(quats)

    def quat_to_euler_pose(self, pose):
        # print("pose"+str(pose))
        if len(pose.shape) > 2:
            rots = self.euler_from_quaternion(pose[:, :, 3:-1])
            pose_euler = np.concatenate([pose[:, :, 0:3], rots, np.expand_dims(pose[:, :, -1], -1)], axis=-1)
        else:
            rots = self.euler_from_quaternion(pose[:, 3:-1])
            pose_euler = np.concatenate([pose[:, 0:3], rots, np.expand_dims(pose[:, -1], -1)], axis=-1)
        return pose_euler

    def euler_to_quat_pose(self, pose):
        if len(pose.shape) > 2:
            quats = self.quaternion_from_euler(pose[:, :, 3:-1])
            pose_euler = np.concatenate([pose[:, :, 0:3], quats, np.expand_dims(pose[:, :, -1], -1)], axis=-1)
        else:
            quats = self.quaternion_from_euler(pose[:, 3:-1])
            pose_euler = np.concatenate([pose[:, 0:3], quats, np.expand_dims(pose[:, -1], -1)], axis=-1)
        return pose_euler

    def normalize_actions(self, actions):
        actions[:, 0:3] = (actions[:, 0:3] - self.act_mins[0:3]) / (self.action_range[0:3])
        actions[:, 3:-1] = (actions[:, 3:-1] - self.euler_mins) / (self.euler_range)
        return 2 * actions - 1

    def normalize_ee_poses(self, ee_poses):
        ee_poses[:, :, :-1] = self.normalize_raw_pose(ee_poses[:, :, :-1])
        return 2 * ee_poses - 1

    def normalize_raw_pose(self, pose):
        pose[:, :, 0:3] = self.normalize_xyz_only(pose[:, :, 0:3])
        pose[:, :, 3:] = (pose[:, :, 3:] - self.euler_mins) / (self.euler_range)
        return pose

    def normalize_xyz_only(self, xyz):
        xyz = (xyz - self.ee_mins[0:3]) / (self.ee_range[0:3])
        return xyz

    def normalize_bbox(self, bbox):
        bbox[:2] = (bbox[:2] - self.ee_mins[0]) / (self.ee_range[0])
        bbox[2:4] = (bbox[2:4] - self.ee_mins[1]) / (self.ee_range[1])
        bbox[4:6] = (bbox[4:6] - self.ee_mins[2]) / (self.ee_range[2])
        return bbox

    def denormalize_ee_poses(self, ee_poses):
        # ee_poses[:, :, 0:3] = (ee_poses[:, :, 0:3] - self.ee_mins[0:3]) / (self.ee_range[0:3])
        # ee_poses[:, :, 3:-1] = (ee_poses[:, :, 3:-1] - self.euler_mins) / (self.euler_range)
        ee_poses = (ee_poses + 1) / 2
        ee_poses[:, :, 0:3] = ee_poses[:, :, 0:3] * self.ee_range[0:3] + self.ee_mins[0:3]
        ee_poses[:, :, 3:-1] = ee_poses[:, :, 3:-1] * self.euler_range + self.euler_mins
        return ee_poses

    def denormalize_xyz_poses(self, ee_poses):
        # ee_poses = (ee_poses+1)/2
        ee_poses[:, :, 0:3] = ee_poses[:, :, 0:3] * (self.ee_range[0:3])  # + self.ee_mins[0:3]
        return ee_poses

    def normalize_images(self, images):
        images = images / 255
        return images

    def select_ts_btw_waypoints(self, waypoints):
        fut_wp = waypoints[1:]
        past_wp = waypoints[:-1]
        delta_wp = fut_wp - past_wp
        # delta_ts = np.repeat(np.expand_dims(delta_wp//(self.ts_ov_wp_ratio+1), axis=1), self.ts_ov_wp_ratio, axis=1)
        # delta_ts = np.random.randint(past_wp, high=fut_wp, size=(past_wp.shape[0], self.ts_ov_wp_ratio))
        # all_ts = np.repeat(np.expand_dims(np.arange(self.ts_ov_wp_ratio)+1, axis=0), waypoints.shape[0]-1, axis=0)
        # all_ts = np.multiply(delta_ts, all_ts)+np.repeat(np.expand_dims(past_wp, axis=1), self.ts_ov_wp_ratio, axis=1)
        all_ts = np.zeros((waypoints.shape[0] + (waypoints.shape[0] - 1) * self.ts_ov_wp_ratio))
        for k in range(waypoints.shape[0] - 1):
            all_ts[k * (1 + self.ts_ov_wp_ratio)] = past_wp[k]
            # print(np.random.choice(np.arange(past_wp[k], fut_wp[k]), size=(self.ts_ov_wp_ratio)))
            available_ts = np.arange(past_wp[k] + 1, fut_wp[k]) if (past_wp[k] + 1) < fut_wp[k] else np.arange(
                past_wp[k], fut_wp[k])
            if self.ts_ov_wp_ratio > 1:
                all_ts[
                k * (1 + self.ts_ov_wp_ratio) + 1:k * (1 + self.ts_ov_wp_ratio) + self.ts_ov_wp_ratio + 1] = np.sort(
                    np.random.choice(available_ts, size=(self.ts_ov_wp_ratio)))
            else:
                all_ts[k * (1 + self.ts_ov_wp_ratio) + 1:k * (
                            1 + self.ts_ov_wp_ratio) + self.ts_ov_wp_ratio + 1] = np.random.choice(available_ts, size=(
                    self.ts_ov_wp_ratio))
        all_ts[-1] = waypoints[-1]
        return all_ts

    def select_local_ts_from_wp(self, waypoints):
        loc = np.arange(self.n_local_ts)
        waypoints = np.repeat(np.expand_dims(waypoints, axis=len(waypoints.shape)), self.n_local_ts,
                              axis=len(waypoints.shape))
        waypoints = waypoints - np.flip(loc)
        return waypoints

    def get_min_max_waypoint(self):
        tasks = self.config["train_task_ids"]
        wp_lens = []
        for task_i, task in enumerate(tasks):
            for j in range(self.config["n_variation_per_task"]):
                for k in range(self.config["n_ex_per_variation"]):
                    wp = np.load(self.path_dataset + "train/" + str(task) + "/obs/" + self.used_cams[0] + "/" + str(
                        j) + "/actual_waypoints_ts_" + str(k) + ".npy")
                    wp_lens.append(wp.shape[0])
        #print(np.max(np.array(wp_lens)))
        #print(np.min(np.array(wp_lens)))
        return np.min(np.array(wp_lens)), np.max(np.array(wp_lens))

    def load_traj(self, task, variation, exemple, ee_poses, actions, obs, all_waypoints, n_wps, batch_id, obj_poses, demo_id=-1, training=True):
        hist_action_path = self.path_dataset + "train/" + str(task) + "/actions/gripper_pose/" + str(
            variation) + "/" + str(exemple) + ".npy"
        hist_ee_path = self.path_dataset + "train/" + str(task) + "/ee_state/" + str(variation) + "/" + str(
            exemple) + ".npy"
        waypoints_path = self.path_dataset + "train/" + str(task) + "/obs/" + self.used_cams[0] + "/" + str(
            variation) + "/actual_waypoints_ts_" + str(exemple) + ".npy"
        obj_pos_path = self.path_dataset + "train/" + str(task) + "/obj_poses/" + str(variation) + "/" + str(
            exemple) + ".json"

        # act = np.load(hist_action_path)
        act = np.load(hist_action_path, allow_pickle=True)
        # act = np.array((act.item())['gripper_pose'])

        ee_pos = np.load(hist_ee_path)
        waypoints = np.load(waypoints_path)
        if waypoints[-1] != (ee_pos.shape[0] - 1):
            waypoints = np.concatenate([waypoints, np.array([ee_pos.shape[0] - 1])])
        obj_pos_fp = open(obj_pos_path, 'r')
        ob_pos = json.load(obj_pos_fp)

        ee_pos = np.concatenate([np.zeros((self.n_local_ts - 1, ee_pos.shape[-1])), ee_pos])

        # print(ee_pos)
        if demo_id == -1:
            all_glob_ts = self.select_ts_btw_waypoints(waypoints).astype(int)
        else:
            all_glob_ts = waypoints.astype(int)
        all_ts = (self.select_local_ts_from_wp(all_glob_ts) + (self.n_local_ts - 1))
        # print(all_ts)

        """if (not training) and (demo_id == -1):
            timesteps_considered = 1
        else:"""
        timesteps_considered = all_ts.shape[0]
        #print(ob_pos)
        #print(all_ts)
        #print("all_ts : "+str(all_ts))
        for k_g in range(len(all_ts)):
            for k_l in range(len(all_ts[0])):
                #if k_g > 0 or k_l > self.n_local_ts - 1:
                ts = all_ts[k_g][k_l] - (self.n_local_ts - 1)
                if ts>=0:
                    #print("ts : "+str(ts))
                    #print("(self.n_local_ts - 1) : " + str((self.n_local_ts - 1)))
                    n_obj = 0
                    for obj in ob_pos[str(ts)]:
                        normed_pose = 2*self.normalize_bbox(np.array(obj["bbox"]))-1
                        concat_obj_infos = np.concatenate([indices_to_one_hot(int(self.object_names_to_id[obj["name"]]), self.n_obj_classes).flatten(), normed_pose])
                        if demo_id == -1:
                            obj_poses[batch_id, k_g, k_l, n_obj, :] = concat_obj_infos
                        else:
                            obj_poses[batch_id, demo_id, k_g, k_l, n_obj, :] = concat_obj_infos
                        n_obj += 1
        #print("obj_poses[batch_id, demo_id] : "+str(obj_poses[batch_id, demo_id]))
        """for k, ts in all_ts:
            for obj in ob_pos[ts]:
                if demo_id == -1:
                    obj_poses[batch_id, demo_id, :all_ts.shape[0], :, :]
                else:
                    obj_poses[]"""

        if demo_id == -1:
            all_waypoints[batch_id, :timesteps_considered, :] = all_ts[:timesteps_considered, :]  # np.array(waypoints)
            n_wps[batch_id] = waypoints.shape[0]
        else:
            all_waypoints[batch_id, demo_id, :all_ts.shape[0], :] = all_ts  # np.array(waypoints)
            n_wps[batch_id, demo_id] = waypoints.shape[0]

        ee_pos = ee_pos[all_ts]#self.quat_to_euler_pose(ee_pos[all_ts])#self.normalize_ee_poses(self.quat_to_euler_pose(ee_pos[all_ts]))
        ee_pos[:, :, 0:3] = self.normalize_xyz_only(ee_pos[:, :, 0:3])
        act = act[all_glob_ts]#self.quat_to_euler_pose(act[all_glob_ts])#self.normalize_actions(self.quat_to_euler_pose(act[all_glob_ts]))
        act[:, 0:3] = self.normalize_xyz_only(act[:, 0:3])

        if demo_id == -1:
            ee_poses[batch_id, :timesteps_considered, :, :] = ee_pos[:timesteps_considered, :]
        else:
            ee_poses[batch_id, demo_id, :all_ts.shape[0], :, :] = ee_pos

        if demo_id == -1:
            actions[batch_id, :timesteps_considered, :] = act[:timesteps_considered, :]
        else:
            actions[batch_id, demo_id, :all_glob_ts.shape[0], :] = act

        for cam_id, cam in enumerate(self.used_cams):
            hist_path = self.path_dataset + "train/" + str(task) + "/obs/" + cam + "/" + str(variation) + "/" + str(
                exemple) + "_rgb.mp4"
            cap = cv2.VideoCapture(hist_path)
            frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
            fc = 0
            ret = True
            """if (not training) and (demo_id == -1):
                frameCount=1"""
            while (fc < frameCount and ret):
                ret, buf[fc] = cap.read()
                fc += 1
            cap.release()
            padding = np.zeros((self.n_local_ts - 1,) + buf.shape[-3:])
            # print(padding.shape)
            hist_ = np.concatenate([padding, buf])
            # print(hist_)
            # input()
            hist_ = np.transpose(hist_, (0, 3, 1, 2))
            if demo_id == -1:
                obs[batch_id, :all_ts.shape[0], :, cam_id, :, :, :] = self.normalize_images(hist_[all_ts])
            else:
                obs[batch_id, demo_id, :all_ts.shape[0], :, cam_id, :, :, :] = self.normalize_images(hist_[all_ts])
        return ee_poses, actions, obs, all_waypoints, n_wps, obj_poses

    def load_batch_by_waypoints(self, demo_only=False):
        if self.current_batch >= self.n_batches:
            self.prepare_epoch()
        batch = self.all_exemples[self.batches_ids[self.current_batch]]
        self.current_batch += 1
        # print(self.n_epochs)
        # print(batch)
        n_tot_ts = self.n_global_ts + (self.n_global_ts - 1) * self.ts_ov_wp_ratio
        all_waypoints = np.zeros((self.batch_size, n_tot_ts, self.n_local_ts)) + float('-inf')
        all_waypoints_demos = np.zeros((self.batch_size, self.n_demos, self.n_global_ts, self.n_local_ts)) + float(
            '-inf')

        if self.get_depth:
            demos = np.zeros((self.batch_size, self.n_demos, self.n_global_ts, self.n_local_ts, self.n_cameras, 4, 128,
                              128))  # .to(device)
            history = np.zeros((self.batch_size, n_tot_ts, self.n_local_ts, self.n_cameras, 4, 128, 128))  # .to(device)
        else:
            demos = np.zeros((self.batch_size, self.n_demos, self.n_global_ts, self.n_local_ts, self.n_cameras, 3, 128,
                              128))  # .to(device)
            history = np.zeros((self.batch_size, n_tot_ts, self.n_local_ts, self.n_cameras, 3, 128, 128))  # .to(device)

        ee_poses = np.zeros((self.batch_size, n_tot_ts, self.n_local_ts, self.config["ee_state_dim"]))
        actions = np.zeros((self.batch_size, n_tot_ts, self.config["ee_state_dim"]))
        obj_poses = np.zeros((self.batch_size, n_tot_ts, self.n_local_ts, self.n_obj, self.low_dim))

        ee_poses_demos = np.zeros((self.batch_size, self.n_demos, self.n_global_ts, self.n_local_ts, self.config["ee_state_dim"]))
        actions_demos = np.zeros((self.batch_size, self.n_demos, self.n_global_ts, self.config["ee_state_dim"]))
        obj_poses_demos = np.zeros((self.batch_size, self.n_demos, self.n_global_ts, self.n_local_ts, self.n_obj, self.low_dim))

        traj_n_wps = np.zeros((self.batch_size))
        demos_n_wps = np.zeros((self.batch_size, self.n_demos))

        batch_id = 0
        batch_infos = []
        for task, variation, exemple in batch:
            variations_path = self.path_dataset + "train/" + str(task) + "/variation_numbers.npy"
            variation_idx = np.load(variations_path)[variation]
            batch_infos.append((task, variation_idx))
            # demos_idx = np.setdiff1d(np.arange(self.n_demos+1), np.array([exemple]))
            demos_idx = np.setdiff1d(np.arange(self.config["n_ex_per_variation"]), np.array([exemple]))
            demos_idx = np.random.choice(demos_idx, self.n_demos)
            if not demo_only:
                ee_poses, actions, history, all_waypoints, traj_n_wps, obj_poses = self.load_traj(task, variation,
                                                                                                  exemple, ee_poses,
                                                                                                  actions, history,
                                                                                                  all_waypoints,
                                                                                                  traj_n_wps, batch_id,
                                                                                                  obj_poses)
            for k_i, k in enumerate(demos_idx):
                ee_poses_demos, actions_demos, demos, all_waypoints_demos, demos_n_wps, obj_poses_demos = self.load_traj(
                    task, variation, k, ee_poses_demos, actions_demos, demos, all_waypoints_demos, demos_n_wps,
                    batch_id, obj_poses_demos, demo_id=k_i)
            batch_id += 1
        training_batch = {"traj_ee_poses": ee_poses, "traj_actions": actions, "traj_imgs": history,
                          "traj_timesteps": all_waypoints, "traj_obj_poses": obj_poses, "traj_n_waypoints": traj_n_wps,
                          "demos_ee_poses": ee_poses_demos, "demos_actions": actions_demos, "demos_imgs": demos,
                          "demos_timesteps": all_waypoints_demos, "demos_n_waypoints": demos_n_wps,
                          "demos_obj_poses": obj_poses_demos}
        return training_batch, np.array(batch_infos)

    def prepare_epoch(self):
        self.n_batches = self.train_set_size // self.batch_size
        self.batches_ids = np.random.choice(self.all_exemples_ids, size=(self.n_batches, self.batch_size),
                                            replace=False)
        self.current_batch = 0
        self.n_epochs += 1

    def init_infer_traj_from_train_set(self):
        batch, batch_infos = self.load_batch_by_waypoints(demo_only=True)
        return batch, batch_infos

    def load_robomimic_batch(self):
        """batch.keys() : dict_keys(['actions', 'rewards', 'dones', 'obs', 'next_obs'])

        batch['actions'].shape() : torch.Size([100, 1, 7])

        batch['obs'].keys() : dict_keys(['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'])
        """
        if self.current_batch >= self.n_batches:
            self.prepare_epoch()
        batch = self.all_exemples[self.batches_ids[self.current_batch]]
        self.current_batch += 1
        # print(self.n_epochs)
        # print(batch)
        n_tot_ts = self.n_global_ts + (self.n_global_ts - 1) * self.ts_ov_wp_ratio
        all_waypoints = np.zeros((self.batch_size, n_tot_ts, self.n_local_ts)) + float('-inf')


        if self.get_depth:
            history = np.zeros((self.batch_size, n_tot_ts, self.n_local_ts, self.n_cameras, 4, 128, 128))  # .to(device)
        else:
            history = np.zeros((self.batch_size, n_tot_ts, self.n_local_ts, self.n_cameras, 3, 128, 128))  # .to(device)

        ee_poses = np.zeros((self.batch_size, n_tot_ts, self.n_local_ts, self.config["ee_state_dim"]))
        actions = np.zeros((self.batch_size, n_tot_ts, self.config["ee_state_dim"]))
        obj_poses = np.zeros((self.batch_size, n_tot_ts, self.n_local_ts, self.n_obj, self.low_dim))


        traj_n_wps = np.zeros((self.batch_size))
        demos_n_wps = np.zeros((self.batch_size, self.n_demos))

        batch_id = 0
        batch_infos = []
        for task, variation, exemple in batch:
            variations_path = self.path_dataset + "train/" + str(task) + "/variation_numbers.npy"
            variation_idx = np.load(variations_path)[variation]
            batch_infos.append((task, variation_idx))
            # demos_idx = np.setdiff1d(np.arange(self.n_demos+1), np.array([exemple]))
            demos_idx = np.setdiff1d(np.arange(self.config["n_ex_per_variation"]), np.array([exemple]))
            demos_idx = np.random.choice(demos_idx, self.n_demos)
            if not demo_only:
                ee_poses, actions, history, all_waypoints, traj_n_wps, obj_poses = self.load_traj(task, variation,
                                                                                                  exemple, ee_poses,
                                                                                                  actions, history,
                                                                                                  all_waypoints,
                                                                                                  traj_n_wps, batch_id,
                                                                                                  obj_poses)
            for k_i, k in enumerate(demos_idx):
                ee_poses_demos, actions_demos, demos, all_waypoints_demos, demos_n_wps, obj_poses_demos = self.load_traj(
                    task, variation, k, ee_poses_demos, actions_demos, demos, all_waypoints_demos, demos_n_wps,
                    batch_id, obj_poses_demos, demo_id=k_i)
            batch_id += 1
        training_batch = {"traj_ee_poses": ee_poses, "traj_actions": actions, "traj_imgs": history,
                          "traj_timesteps": all_waypoints, "traj_obj_poses": obj_poses, "traj_n_waypoints": traj_n_wps,
                          "demos_ee_poses": ee_poses_demos, "demos_actions": actions_demos, "demos_imgs": demos,
                          "demos_timesteps": all_waypoints_demos, "demos_n_waypoints": demos_n_wps,
                          "demos_obj_poses": obj_poses_demos}
        return training_batch, np.array(batch_infos)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dataset', type=str, help='path to demos_dataset. ex: ./my_dataset/', required=True)
    parser.add_argument('--path_dataset_config', type=str, help='path to dataset config. ex: ./dataset_config.json', required=True)
    parser.add_argument('--robomimic_name', type=str, help='name of robomimic hdf5 dataset', required=False)
    parser.add_argument('--action_range', type=str, help='next_ts --> actions are next timestep pose, next_wp --> actions are next waypoint pose', required=False)
    parser.add_argument('--visualize', type=bool, required=False)

    args = parser.parse_args()
    # path_dataset = "datasets/4_demos_95_train_5_test/"
    path_dataset = args.path_dataset  # "datasets/4_demos_28_train/"
    file_config = open(args.path_dataset_config, 'r')
    config = json.load(file_config)


    print('What do you want to do : \n')
    print('Generate raw data from RLBench : 1')
    print('Convert raw data to robomimic format : 2')
    print('Get raw data stats (to later normalize models inputs) : 3')
    print('get cameras matrixes : 4')
    print('visualize segmentation colors : 5')
    operation = int(input('Your choice : '))

    if operation==1:
        n_processes = int(input("How much processes to parallelize on : "))
        skip_task_on_var_fail = bool(int(input("skip whole task on variation fail? No:0, Yes:1")))
        max_traj_trials = int(input("number of trajectory trials before declaring var fail : "))
        try:
            os.mkdir(path_dataset)
        except:
            print('Creation of directory "' + str(path_dataset) + '" failed! If a directory with the same name already exists, please choose an other name or delete the existing one.')
            sys.exit("Ending process")

        try:
            os.mkdir(path_dataset + "train/")
            os.mkdir(path_dataset + "test/")
        except:
            print('A problem occured while creating train/test dirs!')
            sys.exit("Ending process")

        mode = "train"
        train_tasks = FS95_V1['train']
        test_tasks = FS95_V1['test']
        if mode == 'train':
            if config['train_task_ids'][0] == "*":
                n_tasks = len(train_tasks)
                tasks_per_processes = n_tasks//n_processes
                rest = n_tasks%n_processes
                asssigned_tasks = [k*tasks_per_processes+np.arange(tasks_per_processes) for k in range(n_processes)]
                last_process = np.arange(rest)+(tasks_per_processes*n_processes)
                asssigned_tasks.append(last_process)
                """print(asssigned_tasks)
                input()"""
            else:
                n_tasks = len(config['train_task_ids'])
                print(n_tasks)
                tasks_per_processes = n_tasks // n_processes
                print(tasks_per_processes)
                rest = n_tasks % n_processes
                print(rest)
                asssigned_tasks = [k * tasks_per_processes + np.arange(tasks_per_processes) for k in range(n_processes)]
                print(asssigned_tasks)
                last_process = np.array(np.arange(rest) + (tasks_per_processes * n_processes))
                print(last_process)
                asssigned_tasks.append(last_process)
                print(asssigned_tasks)
                act_tasks = np.array(config['train_task_ids'])
                asssigned_tasks = [act_tasks[np.array(asssigned_tasks[i])] for i in range(len(asssigned_tasks))]#np.array(config['train_task_ids'])[np.array(asssigned_tasks)]
        else:
            if config['test_task_ids'][0] == "*":
                n_tasks = len(test_tasks)
                tasks_per_processes = n_tasks // n_processes
                rest = n_tasks % n_processes
                asssigned_tasks = [k * tasks_per_processes + np.arange(tasks_per_processes) for k in range(n_processes)]
                last_process = np.arange(rest) + (tasks_per_processes * n_processes)
                asssigned_tasks.append(last_process)
                """print(asssigned_tasks)
                input()"""
            else:
                n_tasks = len(config['test_task_ids'])
                print(n_tasks)
                tasks_per_processes = n_tasks // n_processes
                print(tasks_per_processes)
                rest = n_tasks % n_processes
                print(rest)
                asssigned_tasks = [k * tasks_per_processes + np.arange(tasks_per_processes) for k in range(n_processes)]
                print(asssigned_tasks)
                last_process = np.array(np.arange(rest) + (tasks_per_processes * n_processes))
                print(last_process)
                asssigned_tasks.append(last_process)
                print(asssigned_tasks)
                act_tasks = np.array(config['test_task_ids'])
                asssigned_tasks = [act_tasks[np.array(asssigned_tasks[i])] for i in range(
                    len(asssigned_tasks))]  # np.array(config['test_task_ids'])[np.array(asssigned_tasks)]

        waypoints_demo_generator = Waypoints_demo_generator(path_dataset, config, mode=mode, skip_task_on_var_fail=skip_task_on_var_fail, max_traj_trials=max_traj_trials)
        print(asssigned_tasks)
        with Pool(n_processes+1) as p:
            print(p.map(waypoints_demo_generator.get_demos, asssigned_tasks))

        with open(path_dataset + 'dataset_config.json', 'w') as f:
            json.dump(config, f)

    elif operation==2:
        #export_dataset_to_pose_estimation_data(path_dataset, config, action_range=args.action_range, visited_states='all'
        export_dataset_to_keypoints_estimation_data(path_dataset, config, action_range=args.action_range, visited_states='all', visualize_data=args.visualize)

    elif operation==3:
        if config['train_task_ids'][0] == "*":
            mode = "train"
            train_tasks = FS95_V1['train']
            test_tasks = FS95_V1['test']
            if mode == 'train':
                n_tasks = len(train_tasks)
                config['train_task_ids'] = np.arange(n_tasks).tolist()
            else:
                n_tasks = len(test_tasks)
                config['test_task_ids'] = np.arange(n_tasks).tolist()


        get_dataset_stats(path_dataset, config)
        filenames = get_all_objects_names(path_dataset, config, mode="train")
        remove_duplicate_object_names(filenames)
    elif operation==4:
        if config['train_task_ids'][0] == "*":
            mode = "train"
            train_tasks = FS95_V1['train']
            test_tasks = FS95_V1['test']
            if mode == 'train':
                n_tasks = len(train_tasks)
                config['train_task_ids'] = np.arange(n_tasks).tolist()
            else:
                n_tasks = len(test_tasks)
                config['test_task_ids'] = np.arange(n_tasks).tolist()
        get_cameras_matrixes(config)
    elif operation==5:

        if config['train_task_ids'][0] == "*":
            mode = "train"
            train_tasks = FS95_V1['train']
            test_tasks = FS95_V1['test']
            if mode == 'train':
                n_tasks = len(train_tasks)
                config['train_task_ids'] = np.arange(n_tasks).tolist()
            else:
                n_tasks = len(test_tasks)
                config['test_task_ids'] = np.arange(n_tasks).tolist()
        print("Availbale tasks ids : " +str(config['train_task_ids']))
        task = int(input("Id of Task to consider : "))
        visualize_mask_colors(path_dataset, config, tasks = [task], action_range=args.action_range, visited_states='all')
    #get_dataset_stats(path_dataset, config)

    #
    #
    # get_all_objects_names()
    # remove_duplicate_object_names("all_RLBench_objects_names.txt")
    # load_objects_names("all_RLBench_objects_names_cleaned.txt")
    """path_dataset = args.path_dataset
    rlbench_dataset = Dataset(path_dataset, 8, 2, ["left_shoulder", "right_shoulder"], 5, 8, depth=False)
    rlbench_dataset.init_infer_traj_from_train_set()"""
    """for k in range(80):
        print("k : "+str(k))
        rlbench_dataset.load_batch_by_waypoints()"""

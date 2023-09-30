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



class Demo_Cond_BC_Test():
    def __init__(self, dataset_config, train_config, infer_train_tasks=False):
        self.dataset_config = dataset_config
        obs_config = ObservationConfig(image_size=(dataset_config["obs_dim"][0], dataset_config["obs_dim"][1]))
        obs_config.set_all(True)
        action_mode = ActionMode(eval("ArmActionMode." + str(dataset_config['action_mode'])))
        self.env = Environment(
            action_mode, obs_config=obs_config, headless=dataset_config['headless_generation'])
        self.env.launch()
        self.cameras = train_config['cams']
        self.train_config = train_config

    def setup_task(self, task, variation):
        self.task = self.env.get_task(task)
        self.task._variation_number = variation

    def process_obs(self, obs):
        fps = 10
        rgb_cams = []
        depth_cams = []
        for cam in self.cameras:
            cam_rgb = cam + "_rgb"
            cam_depth = cam + "_depth"
            rgb_cams.append(np.array(eval('obs.' + str(cam_rgb))).astype(np.uint8))
            # depth = np.expand_dims(np.array(eval('raw_demos[d][t].' + str(cam_depth))), 0)
            if self.dataset_config["get_depth"]:
                depth_cams.append((np.array(eval('obs.' + str(cam_depth))) * 256).astype(np.uint8))
            else:
                depth_cams.append(None)
        ee_state = np.concatenate([np.array(obs.gripper_pose), np.array([obs.gripper_open])], axis=-1)
        return rgb_cams, depth_cams, ee_state

    def reset_task(self):
        descriptions, obs = self.task.reset()
        rgb, depth, ee_state = self.process_obs(obs)
        return {"obs":{"rgb":rgb, "depth":depth}, "ee_state" : ee_state}

    def step(self, action, n_obs):
        all_rgb = []
        all_depth = []
        all_ee = []
        obs, reward, terminate = self.task.step_several_ts_obs(action, n_obs)
        for k in range(len(obs)):
            rgb, depth, ee_state = self.process_obs(obs[k])
            all_rgb.append(rgb)
            all_depth.append(depth)
            all_ee.append(ee_state)
        return {"obs": {"rgb": all_rgb, "depth": all_depth}, "ee_state": all_ee, "terminate":terminate}


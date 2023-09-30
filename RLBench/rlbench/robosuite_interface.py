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
#from PIL import Image
#import cv2
import itertools
from scipy.spatial.transform import Rotation as R
import robomimic.envs.env_base as EB


class Robosuite_Interface(EB.EnvBase):
    def __init__(self,
        env_name,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
        postprocess_visual_obs=True,
        **kwargs):
        print(kwargs)
        task_ids = kwargs["task_ids"]
        if len(task_ids)>1:
            print("multi-task not implemented yet")
        tsk = task_ids[0]
        action_range = kwargs["action_range"]
        robomimic_name = kwargs["robomimic_name"]
        file_config = open('../exps/RAL/'+robomimic_name+'/'+str(action_range)+"/"+str(tsk)+'/noise/bc/dataset_config.json', 'r')
        dataset_config = json.load(file_config)

        self.ee_maxs = np.load('../exps/RAL/'+robomimic_name+'/'+str(action_range)+"/"+str(tsk)+'/noise/bc/' + "ee_maxs.npy")
        self.ee_mins = np.load('../exps/RAL/'+robomimic_name+'/'+str(action_range)+"/"+str(tsk)+'/noise/bc/' + "ee_mins.npy")
        self.ee_ranges = self.ee_maxs - self.ee_mins

        self.obj_pose_maxs = np.load('../exps/RAL/' + robomimic_name + '/' + str(action_range) + "/" + str(tsk) + '/noise/bc/' + "obj_pose_maxs.npy")
        self.obj_pose_mins = np.load('../exps/RAL/' + robomimic_name + '/' + str(action_range) + "/" + str(tsk) + '/noise/bc/' + "obj_pose_mins.npy")
        self.obj_pose_ranges = self.obj_pose_maxs - self.obj_pose_mins

        #super(Robosuite_Interface, self).__init__(env_name)
        self._env_name = env_name
        self.dataset_config = dataset_config
        print(dataset_config['action_mode'])
        obs_config = ObservationConfig(image_size=(dataset_config["obs_dim"][0], dataset_config["obs_dim"][1]))
        obs_config.set_all(True)
        action_mode = ActionMode(eval("ArmActionMode." + str(dataset_config['action_mode'])))
        print("1")
        self.env = Environment(action_mode, obs_config=obs_config, headless=dataset_config['headless_generation'], enable_path_observations=False)
        print("2")
        self.env.launch()
        print("3")
        self.cameras = ["left_shoulder", "right_shoulder", "wrist", "overhead", "front"]#dataset_config['cams']
        #self.train_config = train_config
        """self.object_names_to_id = self.load_objects_names("../exps/templates/all_RLBench_objects_names_cleaned_0.txt")
        file_obj_poses = open( "../exps/templates/nb_ob_per_img_0.json", 'r')
        obj_poses = json.load(file_obj_poses)
        self.n_obj_per_img = obj_poses["nb_ob_per_img_max"]"""

        self.object_names_to_id = {}
        self.n_obj_per_img = 0
        for task in dataset_config["train_task_ids"]:
            self.object_names_to_id = self.load_objects_names('../exps/RAL/'+robomimic_name+'/'+str(action_range)+"/"+str(tsk)+'/noise/bc/all_RLBench_objects_names_cleaned.txt', old_dict=self.object_names_to_id)
            file_obj_poses = open('../exps/RAL/'+robomimic_name+'/' + str(action_range)+"/" + str(tsk) + '/noise/bc/nb_ob_per_img.txt', 'r')
            variation_numbers = np.load('../exps/RAL/'+robomimic_name+'/' + str(action_range) + "/" + str(tsk) + '/noise/bc/variation_numbers.npy', 'r')
            obj_poses = json.load(file_obj_poses)
            self.n_obj_per_img = max(obj_poses["nb_ob_per_img_max"], self.n_obj_per_img)


        self.n_obj_classes = len(self.object_names_to_id) + 1  # +1 for gripper
        #self.low_dim = (6 + self.n_obj_classes)
        if self.dataset_config['use_object_quaternions']:
            self.low_dim = (4 + 6 + 1)
        else:
            self.low_dim = (6 + 1)
        train_tasks = FS95_V1['train']
        test_tasks = FS95_V1['test']
        sim_tasks = np.array(train_tasks)[dataset_config['train_task_ids']].tolist()
        print(np.random.choice(sim_tasks))
        print(np.random.choice(np.arange(dataset_config['n_variation_per_task'])))
        self.setup_task(np.random.choice(sim_tasks), variation_numbers[0])

    def load_objects_names(self, obj_names_file, old_dict=None):
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

    def setup_task(self, task, variation):
        self.task = self.env.get_task(task)
        self.task._variation_number = variation

    def indices_to_one_hot(self, data, nb_classes):
        """Convert an iterable of indices to one-hot encoded labels."""
        targets = np.array(data).reshape(-1)
        return np.eye(nb_classes)[targets]

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
        ee_state = (np.concatenate([np.array(obs.gripper_pose), np.array([obs.gripper_open])], axis=-1)-self.ee_mins)/self.ee_ranges
        ee_euler_state = np.concatenate([np.array(obs.gripper_euler_pose), np.array([obs.gripper_open])], axis=-1)

        n_obj = 0
        concat_obj_infos = np.array([])
        traj_obj_poses = np.zeros((self.n_obj_per_img * self.low_dim))
        #traj_obj_poses = np.zeros((self.n_obj_per_img, self.low_dim))

        for obj in obs.obj_annotations['objects']:
            normed_pose = (np.array(obj["bbox"])-self.obj_pose_mins)/self.obj_pose_ranges  # 2 * normalize_bbox(np.array(obj["bbox"])) - 1
            # print('normed_pose : ' + str(normed_pose))
            if self.dataset_config['use_object_quaternions']:
                normed_quat = np.array(obj["quat"])
                # print('normed_quat : ' + str(normed_quat))
                normed_pose = np.concatenate([normed_pose, normed_quat], axis=-1)
            #concat_obj_infos = np.concatenate([self.indices_to_one_hot(int(self.object_names_to_id[obj["name"]]), self.n_obj_classes).flatten(),normed_pose], axis=-1)
            concat_obj_infos = np.concatenate([np.array([float(self.object_names_to_id[obj["name"]]) / self.n_obj_classes]), normed_pose], axis=-1)
            traj_obj_poses[n_obj * self.low_dim:(n_obj + 1) * self.low_dim] = concat_obj_infos
            #traj_obj_poses[n_obj, :] = concat_obj_infos
            n_obj += 1

        # concat_obj_infos = np.concatenate([self.indices_to_one_hot(int(self.object_names_to_id[obj["name"]]), self.n_obj_classes).flatten(), normed_pose], axis=-1)
        return rgb_cams, depth_cams, ee_state, traj_obj_poses, ee_euler_state

    def get_obs_dict(self, ee_state, obj_poses):
        """"""
        obs_dict = {}
        obs_dict["robot0_eef_pos"] = ee_state[:3]
        obs_dict["robot0_eef_quat"] = ee_state[3:-1]
        obs_dict["robot0_gripper_qpos"] = np.array([ee_state[-1]])
        obs_dict["object"] = obj_poses
        """for k in range(obj_poses.shape[0]):
            obs_dict["object_" + str(k)] = obj_poses[k]"""
        return obs_dict


    def reset(self):
        #print("reseting")
        self.successive_action_fails = 0
        self.terminate = False
        descriptions, obs = self.task.reset()
        waypoints = self.task._scene._active_task.get_waypoints()
        self.all_waypoints = []
        for wp in waypoints:
            wp_position = wp._waypoint.get_position()
            wp_orientation = wp._waypoint.get_orientation()
            wp_gripper = [1.0]
            if 'open_gripper(' in wp._waypoint.get_extension_string():
                wp_gripper = [1.0]
            elif 'close_gripper(' in wp._waypoint.get_extension_string():
                wp_gripper = [0.0]
            print("wp_position"+str(wp_position))
            print("wp_orientation"+str(wp_orientation))
            self.all_waypoints.append(np.concatenate([wp_position, wp_orientation, wp_gripper]))
        #print(waypoints)
        #print("reset obs : "+str(obs))
        rgb, depth, ee_state, obj_poses, ee_euler_state = self.process_obs(obs)
        self.ee_euler_states = [ee_euler_state]

        print("ee_state" + str(ee_state))
        self.initial_ee_state = ee_state
        #print(obj_poses)
        obs_dict = self.get_obs_dict(ee_state, obj_poses)
        self.last_obs, self.reward, self.last_terminate = obs, 0.0, self.terminate
        print("---------------- reset ----------------")
        print("obs_dict : " + str(obs_dict))
        return obs_dict
        #return {"obs":{"rgb":rgb, "depth":depth}, "ee_state" : ee_state}

    def get_observation(self, obs=None):
        if obs != None:
            return obs
        else:
            obs = self.task._scene.get_observation()
            rgb, depth, ee_state, obj_poses, ee_euler_state = self.process_obs(obs)
            obs_dict = self.get_obs_dict(ee_state, obj_poses)
            return obs_dict

    def is_success(self):
        success = {'task': self.terminate}
        return success

    def env_meta_data(self):
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
        meta_data = {}
        meta_data["env_name"] = self.dataset_config["name"]
        meta_data["type"] = 1

        env_kwargs = {}
        env_kwargs["has_renderer"] = False
        env_kwargs["has_offscreen_renderer"] = False
        env_kwargs["ignore_done"] = True
        env_kwargs["use_object_obs"] = True
        env_kwargs["controller_configs"] = {}
        env_kwargs["use_camera_obs"] = True
        # env_kwargs["control_freq"] = True
        env_kwargs["camera_depths"] = self.dataset_config["get_depth"]
        env_kwargs["use_object_obs"] = True
        env_kwargs["robots"] = ["Panda"]
        env_kwargs["camera_heights"] = self.dataset_config["obs_dim"][0]
        env_kwargs["camera_widths"] = self.dataset_config["obs_dim"][0]
        # env_kwargs["reward_shaping"] = False
        env_kwargs["camera_names"] = ["front", "wrist"]
        # env_kwargs["render_gpu_device_id"] = 0
        meta_data["env_kwargs"] = env_kwargs
        return meta_data

    def serialize(self):
        return json.dumps(self.env_meta_data())


    def step(self, action):
        all_rgb = []
        all_depth = []
        all_ee = []
        all_obj_poses = []

        #action[3:7]=self.initial_ee_state[3:7]
        #print(action)
        #obs, reward, terminate = self.task.step(action)
        print("step!")
        print("action : " + str(action))
        action = action*self.ee_ranges+self.ee_mins
        action[-1] = max(min(action[-1], 1), 0)

        #action[:3] = action[:3]*self.ee_ranges[:3] + self.ee_mins[:3]
        #action[:3] = action[:3]/10 #* self.ee_ranges + self.ee_mins
        try:
            obs, reward, terminate = self.task.step(action)
            self.last_obs, self.reward, self.last_terminate = obs, reward, terminate
            print("action success")
            self.successive_action_fails = 0
            #print(action)
        except:
            self.successive_action_fails+=1
            print("action failed!")
            #print(action)
            obs, reward, terminate = self.last_obs, self.reward, self.last_terminate
        #obs, reward, terminate = self.task.step(action)
        #self.last_obs, self.reward, self.last_terminate = obs, reward, terminate

        self.terminate = terminate
        #print("obs : "+str(obs))
        #for k in range(len(obs)):
        rgb, depth, ee_state, obj_poses, ee_euler_state = self.process_obs(obs)
        self.ee_euler_states.append(ee_euler_state)
        #print("obj_poses : " + str(obj_poses))

        #all_rgb.append(rgb)
        #all_depth.append(depth)
        #all_ee.append(ee_state)
        #all_obj_poses.append(obj_poses)
        #obs = all_obj_poses[-1]
        obs_dict = self.get_obs_dict(ee_state, obj_poses)
        done = terminate
        info = {"successive_action_fails":self.successive_action_fails, "ee_euler_states":self.ee_euler_states, "all_waypoints":self.all_waypoints}
        print(obj_poses[323:340])
        #return {"obs": {"rgb": all_rgb, "depth": all_depth}, "ee_state": all_ee, "terminate":terminate}
        #print("obs_dict : "+str(obs_dict))
        return obs_dict, reward, done, info




    def get_state(self):
        """
        Get current environment simulator state as a dictionary. Should be compatible with @reset_to.
        """

        return None

    def get_reward(self):
        """
        Get current reward.
        """
        return None

    def get_goal(self):
        """
        Get goal observation. Not all environments support this.
        """
        return None

    def set_goal(self, **kwargs):
        """
        Set goal observation with external specification. Not all environments support this.
        """
        return None

    def is_done(self):
        """
        Check if the task is done (not necessarily successful).
        """

        # Robosuite envs always rollout to fixed horizon.
        return self.terminate


    @property
    def action_dimension(self):
        """
        Returns dimension of actions (int).
        """
        return int(self.dataset_config["action_dim"])

    @property
    def name(self):
        """
        Returns name of environment name (str).
        """
        return self._env_name

    @property
    def type(self):
        """
        Returns environment type (int) for this kind of environment.
        This helps identify this env class.
        """
        return 3



    @classmethod
    def create_for_data_processing(
        cls,
        env_name,
        camera_names,
        camera_height,
        camera_width,
        reward_shaping,
        **kwargs,
    ):
       return None

    @property
    def rollout_exceptions(self):
        """
        Return tuple of exceptions to except when doing rollouts. This is useful to ensure
        that the entire training run doesn't crash because of a bad policy that causes unstable
        simulation computations.
        """
        return None

    def __repr__(self):
        """
        Pretty-print env description.
        """
        return self.name + "\n" #+ json.dumps(self._init_kwargs, sort_keys=True, indent=4)

    def render(self, mode="human", height=None, width=None, camera_name=None, **kwargs):
        obs = self.task._scene.get_observation()
        rgb, depth, ee_state, obj_poses, ee_euler_state = self.process_obs(obs)
        return rgb[3]

    def reset_to(self):
        pass


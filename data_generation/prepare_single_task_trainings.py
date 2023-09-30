import numpy as np
import os
import argparse
import sys
import json
import time
import itertools
import h5py
from multiprocessing import Pool
from dataset_management import  export_dataset_to_robomimic_format
import shutil


#     mkdir 'robomimic/robomimic/exps/RAL/single_task/wp_and_ts/'+str(tsk)+'/noise_or_no_noise/bc_or_cql_.../' for all tsk
#     mkdir "data": "..../robomimic/datasets/RAL/single_task/wp_and_ts/" + str(tsk)+"/data.hdf5" for all tsk
#     mkdir "..../robomimic/trained_models/RAL/single_task/wp_and_ts/" + str(tsk)+"/noise_or_no_noise" for all tsk

# 1 : create dataset_config files for each tsk and paste them to robomimic/robomimic/exps/RAL/single_task/wp_or_ts/'+str(tsk)+'/dataset_config.json'
      # and to path_dataset + "train/" + str(tsk)+"/dataset_config.json"
#     generate h5py dataset from path_dataset + "train/" + str(tsk)+"/dataset_config.json" and path_dataset and save to
      #"..../robomimic/datasets/RAL/single_task/wp_or_ts/data.hdf5"
#   : copy each tasks' all_RLBench_objects_names_cleaned to robomimic/robomimic/exps/RAL/single_task/wp_or_ts/'+str(tsk)+'/all_RLBench_objects_names_cleaned.txt
#   : copy each tasks' nb_ob_per_img.json to robomimic/robomimic/exps/RAL/single_task/wp_or_ts/'+str(tsk)+'/nb_ob_per_img.json

# 2 : create robomimic train config files (.json) with
        #"data": "..../robomimic/datasets/RAL/single_task/wp_or_ts/" + str(tsk)+"/data.hdf5",
        #"output_dir": "..../robomimic/trained_models/RAL/single_task/wp_or_ts/" + str(tsk)+"/noise_or_no_noise/bc_or_cql_.../",
      # and save them to 'robomimic/robomimic/exps/RAL/single_task/wp_and_ts/'+str(tsk)+'/noise_or_no_noise/bc_or_cql_.../'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dataset', type=str, help='path to demos_dataset. ex: ./my_dataset/', required=True)
    parser.add_argument('--path_dataset_config', type=str, help='path to dataset config. ex: ./dataset_config.json',
                        required=True)
    parser.add_argument('--robomimic_name', type=str, help='name of robomimic hdf5 dataset', required=False, default="single_task")
    parser.add_argument('--robomimic_path', type=str, help='path to robomimic code. E.g., /home/lambda-ipi/Desktop/RAL/robomimic/', required=True)
    parser.add_argument('--erase_existing_dirs', type=bool, help='', required=False, default=False)

    args = parser.parse_args()
    # path_dataset = "datasets/4_demos_95_train_5_test/"
    path_dataset = args.path_dataset  # "datasets/4_demos_28_train/"
    erase_existing_dirs = args.erase_existing_dirs

    file_config = open(args.path_dataset_config, 'r')
    config = json.load(file_config)
    tasks_lens={}
    visited_states = 'wp_only'#'all'
    action_ranges = ["next_wp", "next_ts"] if visited_states=='all' else ["next_wp"]
    robomimic_name = args.robomimic_name
    
    
    robomimic_config_templates = {}
    for algo in ["bc", "cql", "bcq", "iris", "hbc", "td3_bc"]:
        robomimic_config_templates[algo] = json.load(open(args.robomimic_path+"robomimic/exps/templates/"+str(algo)+".json" ,'r'))
    single_tsk_path = args.robomimic_path+"robomimic/exps/RAL/"+robomimic_name+"/"
    if os.path.isdir(single_tsk_path):
        if erase_existing_dirs:
            shutil.rmtree(single_tsk_path)
            os.makedirs(single_tsk_path, exist_ok=True)
    else:
        os.makedirs(single_tsk_path, exist_ok=True)
    fails = []
    len_wp_stats = json.load(open(path_dataset + "tasks_wp_lenght_stats.json", 'r'))
    len_stats = json.load(open(path_dataset + "tasks_lenght_stats.json", 'r'))
    if config["train_task_ids"][0]=='*':
        config["train_task_ids"] = np.arange(94).tolist()
    for act_range in action_ranges:
        act_range_path = single_tsk_path+act_range+'/'
        if os.path.isdir(act_range_path):
            if erase_existing_dirs:
                #shutil.rmtree(act_range_path)
                os.mkdir(act_range_path)
        else:
            os.mkdir(act_range_path)
        for tsk in config["train_task_ids"]:
            data_config_ = config.copy()
            data_config_["train_task_ids"] = [tsk]
            data_config_["test_task_ids"] = []
            tsk_path = act_range_path+str(tsk)+'/'
            if os.path.isdir(tsk_path):
                if erase_existing_dirs:
                    #shutil.rmtree(tsk_path)
                    os.mkdir(tsk_path)
            else:
                os.mkdir(tsk_path)
            for noise in ["noise/", "no_noise/"]:
                noise_path = tsk_path+noise
                if os.path.isdir(noise_path):
                    if erase_existing_dirs:
                        #shutil.rmtree(noise_path)
                        os.mkdir(noise_path)
                else:
                    os.mkdir(noise_path)
                for algo in ["bc", "bc_rnn", "cql", "bcq", "hbc", "td3_bc", "iris"]:
                    #try:
                    algo_path = noise_path+algo+'/'
                    if os.path.isdir(algo_path):
                        if erase_existing_dirs:
                            #shutil.rmtree(algo_path)
                            os.mkdir(algo_path)
                    else:
                        os.mkdir(algo_path)
                    if algo == 'bc_rnn':
                        robomimic_config_ = robomimic_config_templates["bc"].copy()
                        robomimic_config_["algo"]["rnn"]["enabled"]=True
                    else:
                        robomimic_config_ = robomimic_config_templates[algo].copy()
                    robomimic_config_["train"]['data']=args.robomimic_path+"datasets/RAL/"+robomimic_name+"/"+act_range+"/" + str(tsk)+"/data.hdf5"
                    robomimic_config_["train"]['output_dir']=args.robomimic_path+"trained_models/RAL/"+robomimic_name+"/"+act_range+"/" + str(tsk)+"/"+noise+"/"+algo+"/"
                    if noise=="noise/":
                        robomimic_config_["observation"]["ee_pose_noise"] = True
                    else:
                        robomimic_config_["observation"]["ee_pose_noise"] = False
                    if act_range=="next_wp":
                        horizon = 7*len_wp_stats[str(tsk)]["max_len"]
                        robomimic_config_["experiment"]["rollout"]["horizon"] = int(horizon)
                    else:
                        horizon = 1.25*len_stats[str(tsk)]["max_len"]
                        robomimic_config_["experiment"]["rollout"]["horizon"] = int(horizon)

                    nb_ob_per_img_path_ = path_dataset + 'train/' + str(tsk) + '/nb_ob_per_img.json'
                    """nb_ob_per_img_ = json.load(open(nb_ob_per_img_path_, 'r'))['nb_ob_per_img_max']
                    objects = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]+["object_" + str(obj_i) for obj_i in range(nb_ob_per_img_)]
                    print(algo)
                    try:
                        robomimic_config_["observation"]["modalities"]["obs"]["low_dim"] = objects
                    except:
                        robomimic_config_["observation"]["planner"]["modalities"]["obs"]["low_dim"] = objects
                        robomimic_config_["observation"]["actor"]["modalities"]["obs"]["low_dim"] = objects"""


                    with open(algo_path + 'robomimic_training_config.json', 'w') as f:
                        json.dump(robomimic_config_, f)
                    with open(algo_path + 'dataset_config.json', 'w') as f:
                        json.dump(data_config_, f)
                    obj_names_path_ = path_dataset + 'train/' + str(tsk) + '/all_RLBench_objects_names_cleaned.txt'
                    nb_ob_per_img_path_ = path_dataset + 'train/' + str(tsk) + '/nb_ob_per_img.json'
                    variation_numbers_path = path_dataset + 'train/' + str(tsk) + '/variation_numbers.npy'
                    ee_maxs_path = path_dataset + 'train/' + str(tsk) + '/ees_maxs.npy'
                    ee_mins_path = path_dataset + 'train/' + str(tsk) + '/ees_mins.npy'
                    obj_pose_maxs_path = path_dataset + 'train/' + str(tsk) + '/obj_pose_maxs.npy'
                    obj_pose_mins_path = path_dataset + 'train/' + str(tsk) + '/obj_pose_mins.npy'
                    os.popen('cp '+obj_names_path_+ ' ' +algo_path+'all_RLBench_objects_names_cleaned.txt')
                    os.popen('cp '+nb_ob_per_img_path_+' '+algo_path +'nb_ob_per_img.txt')
                    os.popen('cp ' + variation_numbers_path + ' ' + algo_path + 'variation_numbers.npy')

                    os.popen('cp ' + ee_maxs_path + ' ' + algo_path + 'ee_maxs.npy')
                    os.popen('cp ' + ee_mins_path + ' ' + algo_path + 'ee_mins.npy')
                    os.popen('cp ' + obj_pose_maxs_path + ' ' + algo_path + 'obj_pose_maxs.npy')
                    os.popen('cp ' + obj_pose_mins_path + ' ' + algo_path + 'obj_pose_mins.npy')
                    #except:
                        #fails.append({'tsk':tsk, 'noise':noise, 'algo':algo, 'act_range':act_range})

    with open('data_preparation_fails.json', 'w') as f:
        json.dump(fails, f)

    data_path = args.robomimic_path + "datasets/RAL/"+robomimic_name+"/"
    if os.path.isdir(data_path):
        if erase_existing_dirs:
            shutil.rmtree(data_path)
            os.makedirs(data_path, exist_ok=True)
    else:
        os.makedirs(data_path, exist_ok=True)
    for act_range in action_ranges:
        act_range_path = data_path+act_range+"/"
        if os.path.isdir(act_range_path):
            if erase_existing_dirs:
                os.mkdir(act_range_path)
        else:
            os.mkdir(act_range_path)
        for tsk in config["train_task_ids"]:
            tsk_path = act_range_path+str(tsk)+'/'
            if os.path.isdir(tsk_path):
                if erase_existing_dirs:
                    os.mkdir(tsk_path)
            else:
                os.mkdir(tsk_path)
            tsk_config = json.load(open(args.robomimic_path+"robomimic/exps/RAL/"+robomimic_name+"/"+act_range+"/"+str(tsk)+"/"+"noise/bc/"+"dataset_config.json", 'r'))
            hdf5_path = tsk_path+"data"
            #try:
            export_dataset_to_robomimic_format(path_dataset, tsk_config, robomimic_path=hdf5_path, robomimic_name = robomimic_name, action_range=act_range, visited_states=visited_states)
            """except:
                print("dataset conversion failed for task "+str(tsk))"""


    output_path = args.robomimic_path + "trained_models/RAL/"+robomimic_name+"/"
    if os.path.isdir(output_path):
        if erase_existing_dirs:
            shutil.rmtree(output_path)
            os.makedirs(output_path, exist_ok=True)
    else:
        os.makedirs(output_path, exist_ok=True)
    for act_range in action_ranges:
        act_range_path = output_path + act_range+ '/'
        if os.path.isdir(act_range_path):
            if erase_existing_dirs:
                os.mkdir(act_range_path)
        else:
            os.mkdir(act_range_path)
        for tsk in config["train_task_ids"]:
            tsk_path = act_range_path + str(tsk) + '/'
            if os.path.isdir(tsk_path):
                if erase_existing_dirs:
                    os.mkdir(tsk_path)
            else:
                os.mkdir(tsk_path)
            for noise in ["noise/", "no_noise/"]:
                noise_path = tsk_path + noise
                if os.path.isdir(noise_path):
                    if erase_existing_dirs:
                        os.mkdir(noise_path)
                else:
                    os.mkdir(noise_path)
                for algo in ["bc", "bc_rnn", "cql", "bcq", "iris", "hbc", "td3_bc"]:
                        algo_path = noise_path + algo + '/'
                        os.mkdir(algo_path)






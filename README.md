# A Keypoints-based Learning Paradigm for Visual Robotic Manipulation
This is the code corresponding to our paper :
 From Temporal-Evolving to Spatial-Fixing: A Keypoints-Based Learning Paradigm for Visual Robotic Manipulation: 
https://ras.papercept.net/conferences/conferences/IROS23/program/IROS23_ContentListWeb_2.html#moat17_11


## Structure of the repository
The code is divided into 3 parts.
The first part is the data generation part. This part allows to (1) generate data from a modified version of the RLBench Simulation, and (2) export the data to a format that can be used to train our 8D waypoints prediction model.
The second part is about training the proposed baseline for 8D waypoints prediction.
The third part allows to evaluate the trained 8D waypoints predictor using RLBench, especially by providing a success rate on the target task.

## Install 

```
# create your virtual environment (tested only with python 3.9)
conda create -n spatial_fixing_il python=3.9
conda activate spatial_fixing_il
```
### RLBench install 
We use a cutom version of RLBench, that allows us to acquire keypoints of interest while generating tasks demonstrations.
RLBench is built around PyRep and V-REP. First head to the 
[PyRep github](https://github.com/stepjam/PyRep) page and install.
**If you previously had PyRep installed, you will need to update your installation!**


Hopefully you have now installed PyRep and have run one of the PyRep examples.
Now lets install our custom RLBench:

```bash
cd RLBench
pip install -r requirements.txt
pip install .
```

### Additionnal dependencies for data generation

go back to the root directory and install requirements:
```bash
cd ..
pip install -r data_gen_requirements.txt
```


### Unsupervised 3D pose Estimation install
```bash
pip install -r  pose_est_requirements.txt
```

## Generate data

### Raw data generation
go to the data generation directory and launch dataset_management.py code, 
```bash
cd data_generation/RLBench
python generate_data.py  --path_dataset data/my_expe/ --path_dataset_config configs/dataset_config.json --n_processes 1
```
, where --path_dataset indicates where the generated data will be stored, and where --path_dataset_config describes some features of the data that needs to be collected. 
Here are the most useful features:
n_ex_per_variation : the number of examples to generate for each task.
headless_generation : whether to lauch the simulation GUI during data generation. GUI considerablyy slows down the data generation. We suggest to only use it for visualization/debug.
"train_task_ids": [0, 8, 41, 79]: defines the tasks for which you want to generate data. The task ids are organized following the order defined in "tasks_ids.txt", where the 0-th line of the file, featuring "reach_target" task corresponds to the task 0, and the 94-th line of the file, featuring task "PourFromCupToCup" correpsonds to task 94. Tasks 9, 12, 40, 54 and 79 correpsonds to the tasks used in our paper.

Once the data generation completed, you'll find raw demonstration data (multi-view image observations, point-clouds, robot states, robot actions, trajectories keypoints, ...) under your "path_dataset".
One more step is required to export data to a format that we can use to train the 8D waypoints prediction models.


### Export data for 8D waypoints prediction model
```bash
export $path_dataset=data/my_expe/
export path_dataset_config=configs/dataset_config.json
sh convert_data_to_3D_pose_estimation.sh
```
The converted data will be automatically placed in the model/data/RLBench_pose_est_data directory

### Training 3D pose estimation models

```bash
cd model
```

training a model on task 0 : 
```bash
python run_h36m_waypoints.py --cfg ./cfg/submit/config_waypoints_task_0.yaml
```

### Monitor training
```bash
cd pose_obj_3d/log/submit/

```

```bash
tensorboard --logdir .  --bind_all
```


### Evaluation of the 3D pose estimation models
! This part is not complete yet !

## Generating a set of test data

## Infering the trained model on the test data
```bash
cd model
python run_h36m_full_pose.py --cfg ./cfg/submit/config_full_pose_test_0.yaml --gpu 0  --eval --test_success_rate --checkpoint ./checkpoint/submit/config_full_pose_0_2023-09-30-18-01/model.bin --gpu 0 --n_frames 1 --eval_batch_size 1 --eval_n_frames 1
```
where "--checkpoint ./checkpoint/submit/task_0/model.bin" indicates the location of the trained model's weights (by default in ./checkpoint/submit)
the predicted 8D waypoints for each examples will be stored in the model/log directory.

## Testing the the predicted 8D waypoints in RLBench
the seeds used to generate each example were save along with the rest of the generated data in "data_generation/data/my_expe/", 
For instance, the seeds for task 0 will be saved in "data_generation/data/my_expe/train/0/seeds".
thanks to these seeds, we can reload the simulation with the same scene configuration used to generate the test examples.
Thereafter, we can execute the corresponding predicted 8D waypoints for each examples, 
finally, we can compute a success rate on these executions.

```bash
 code to compute success rate coming soon!
```

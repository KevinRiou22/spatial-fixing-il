from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget, PlugChargerInPowerSupply, PickAndLift
import numpy as np
import matplotlib.pyplot as plt
from rlbench.tasks import FS95_V1



class ImitationLearning(object):

    def predict_action(self, batch):
        return np.random.uniform(size=(len(batch), 7))

    def behaviour_cloning_loss(self, ground_truth_actions, predicted_actions):
        return 1


# To use 'saved' demos, set the path below, and set live_demos=False
live_demos = True
DATASET = '' if live_demos else 'PATH/TO/YOUR/DATASET'
#random_seed = np.random.get_state()

obs_config = ObservationConfig()
obs_config.set_all(True)

#action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)

#action_mode = ActionMode(ArmActionMode.EE_POSE_EE_FRAME)
#action_mode = ActionMode(ArmActionMode.DELTA_EE_POSE_WORLD_FRAME)
action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME)
env = Environment(
    action_mode, DATASET, obs_config, False)
env.launch()
#train_tasks = FS95_V1['train']
"""for task_i in train_tasks:
    try:
        task = env.get_task(task_i)
        print(len(task._scene._active_task.get_waypoints()))
    except:
        continue"""

task = env.get_task(ReachTarget)#env.get_task(train_tasks[0])
il = ImitationLearning()

demos, timesteps_waypoints = task.get_demos(1, live_demos=live_demos)  # -> List[List[Observation]]
current_seeds = demos[0].random_seed
seeds_f = open("./seeds.txt", 'a')
seeds_f.write(str(current_seeds[0])+","+str(current_seeds[2])+","+str(current_seeds[3])+","+str(current_seeds[4])+"\n")
seeds_f.close()
filename = '0'
np.save(filename, current_seeds[1])
print(current_seeds)
np.random.set_state(current_seeds)
task.reset()
#demos = np.array(demos).flatten()

# An example of using the demos to 'train' using behaviour cloning loss.
# n_iter = 1
# gripper_poses = []
# #print(train_tasks)
# for i in range(n_iter):
#     print("'training' iteration %d" % i)
#     batch = np.random.choice(demos, replace=False)
#     for t in range(len(batch)):
#         gripper_poses.append(np.array(batch[t].gripper_pose))
#         print("t"+str(t)+" :"+str(np.array(batch[t].gripper_pose)))
#     batch_images = [obs.front_point_cloud for obs in batch]
#     print()
#     np.save('pt_cl.npy', batch_images[0])

#
#     input()
#     predicted_actions = il.predict_action(batch_images)
#     ground_truth_actions = [obs.joint_velocities for obs in batch]
#     loss = il.behaviour_cloning_loss(ground_truth_actions, predicted_actions)
# gripper_poses = np.array(gripper_poses)
"""for k in range(gripper_poses.shape[-1]):
    print("min pose component "+str(k)+" : "+str(np.min(gripper_poses[:, k])))
    print("max pose component "+str(k)+" : "+str(np.max(gripper_poses[:, k])))

print('Done')"""
"""ts = -1
demo_id = 0
for k in range(len(timesteps_waypoints[demo_id])):
    ts_wp = timesteps_waypoints[demo_id][k]
    last_sub_demo_img = np.array(demos[demo_id][ts_wp].overhead_rgb)
    plt.imshow(last_sub_demo_img, interpolation='nearest')
    plt.savefig('waypoint_'+str(k)+'_visu.jpg')
    last_sub_demo_img = np.array(demos[demo_id][ts_wp].wrist_rgb)
    plt.imshow(last_sub_demo_img, interpolation='nearest')
    plt.savefig('waypoint_wrist_' + str(k) + '_visu.jpg')
env.shutdown()"""

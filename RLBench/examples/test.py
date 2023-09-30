from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
import numpy as np
from scipy.spatial.transform import Rotation as R


class Agent(object):

    def __init__(self, action_size):
        self.action_size = action_size

    def act(self, obs):
        arm = np.random.normal(0.0, 0.1, size=(self.action_size - 1,))
        gripper = [1.0]  # Always open
        return np.concatenate([arm, gripper], axis=-1)

def euler_from_quaternion(quats):
    eul_rots = []
    for k in range(quats.shape[0]):
        r = R.from_quat(quats[k])
        eul_rots.append(r.as_euler('xyz', degrees=False))
    return np.array(eul_rots)

def quaternion_from_euler(rots):
    quats = []
    for k in range(rots.shape[0]):
        quat_back = R.from_euler('xyz', rots[k], degrees=False)
        quats.append(quat_back.as_quat())
    return np.array(quats)


obs_config = ObservationConfig()
obs_config.set_all(True)

action_mode = ActionMode(ArmActionMode.DELTA_EE_POSE_WORLD_FRAME)
env = Environment(
    action_mode, obs_config=obs_config, headless=False)
env.launch()

task = env.get_task(ReachTarget)

agent = Agent(env.action_size)

descriptions, obs = task.reset()
print(descriptions)
input('send action')
rot = quaternion_from_euler(np.array([[0.0, 0.0, 0.0]]))
pos = np.array([0.0, 0.0, 0.0])

action = np.concatenate([pos, rot[0], np.array([1.0])])

obs, reward, terminate = task.step(action)
input('action sent')
input('send action')
rot = quaternion_from_euler(np.array([[-1.5, 0.0, 0.0]]))
pos = np.array([0.0, 0.0, 0.0])

action = np.concatenate([pos, rot[0], np.array([1.0])])

obs, reward, terminate = task.step(action)
input('action sent')
input('send action')
rot = quaternion_from_euler(np.array([[1.5, 0.0, 0.0]]))
pos = np.array([0.0, 0.0, 0.0])

action = np.concatenate([pos, rot[0], np.array([1.0])])

obs, reward, terminate = task.step(action)
input('action sent')

input('send action')
rot = quaternion_from_euler(np.array([[0.0, 0.0, -1.5]]))
pos = np.array([0.0, 0.0, 0.0])

action = np.concatenate([pos, rot[0], np.array([1.0])])

obs, reward, terminate = task.step(action)
input('action sent')
input('action sent')

input('send action')
rot = quaternion_from_euler(np.array([[0.0, 0.0, 1.5]]))
pos = np.array([0.0, 0.0, 0.0])

action = np.concatenate([pos, rot[0], np.array([1.0])])

obs, reward, terminate = task.step(action)
input('action sent')


print('Done')
env.shutdown()

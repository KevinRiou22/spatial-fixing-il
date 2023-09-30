from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget, PlugChargerInPowerSupply, CloseFridge
import numpy as np
from scipy.spatial.transform import Rotation as R
"""import cv2
import mediapipe as mp
import time



class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z)
                lmlist.append([id, cx, cy, cz])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
        return lmlist"""


class Agent(object):

    def __init__(self, action_size):
        self.action_size = 7
        xmin = 0.4-0.4#-500 / 1000
        xmax = 0.4+0.4#500 / 1000
        ymin = -0.6#-1000 / 1000
        ymax = 0.6#-600 / 1000
        zmin = 0.77#-155 / 1000
        zmax = 1.86#206 / 1000
        rxmin = -3.14159246
        rymin = - 3.14
        rzmin = - 3.14159246
        rxmax = 3.1415919
        rymax = 3.14
        rzmax = 3.14159246

        self.min_dims = np.array([xmin, ymin, zmin, rxmin, rymin, rzmin])
        self.max_dims = np.array([xmax, ymax, zmax, rxmax, rymax, rzmax])
    def euler_from_quaternion(self, quats):
        eul_rots = []
        for k in range(quats.shape[0]):
            r = R.from_quat(quats[k])
            eul_rots.append(r.as_euler('xyz', degrees=False))
        return np.array(eul_rots)

    def quaternion_from_euler(self, rots):
        quats = []
        for k in range(rots.shape[0]):
            quat_back = R.from_euler('xyz', rots[k], degrees=False)
            quats.append(quat_back.as_quat())
        return np.array(quats)

    def act(self, obs):

        arm = np.random.uniform(0.0, 1.0, size=(self.action_size - 1,))
        arm = self.scale_action(arm)
        return arm

    def scale_action(self, action):
        #arm = action * (self.max_dims - self.min_dims) + self.min_dims
        arm = action
        gripper = [1.0]  # Always open
        print(arm)
        arm = np.concatenate([arm[0:3], self.quaternion_from_euler(np.array([arm[3:]])).flatten(), gripper], axis=-1)
        return arm

obs_config = ObservationConfig()
obs_config.set_all(True)
obs_config.gripper_touch_forces = False

action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME)
"""env = Environment(
    action_mode, obs_config=obs_config, headless=False,
    robot_configuration='ur10')"""
env = Environment(
    action_mode, obs_config=obs_config, headless=False,
    robot_configuration='panda')
env.launch()

task = env.get_task(ReachTarget)

agent = Agent(action_size=env.action_size)  # 6DoF + 1 for gripper
#demos = task.get_demos(2, live_demos=True)

training_steps = 120
episode_length = 40
obs = None
descriptions, obs = task.reset()
"""demos, timesteps_waypoints = task.get_demos(2, live_demos=True)

gripper_poses = []

batch = np.random.choice(demos, replace=False)
for t in range(len(batch)):
    gripper_poses.append(np.array(batch[t].gripper_pose))
    print("t"+str(t)+" :"+str(np.array(batch[t].gripper_pose)))
batch_images = [obs.left_shoulder_rgb for obs in batch]

gripper_poses = np.array(gripper_poses)
for k in range(gripper_poses.shape[-1]):
    print("min pose component "+str(k)+" : "+str(np.min(gripper_poses[:, k])))
    print("max pose component "+str(k)+" : "+str(np.max(gripper_poses[:, k])))"""
action = np.concatenate([[0.325, 0, 1.5  ], agent.quaternion_from_euler(np.array([[2, 3.14, 0.0]])).flatten(), [1.0]], axis=-1)
print(action)
obs, reward, terminate = task.step(action)
input("...")
"""for i in range(training_steps):
    if i % episode_length == 0:
        print('Reset Episode')
        descriptions, obs = task.reset()
        print(descriptions)
    try:
        action = agent.act(obs)
        input()
    except:
        continue
    print(action)
    obs, reward, terminate = task.step(action)

print('Done')"""

"""
print("before reset")
descriptions, obs = task.reset()
input()
print("reset done")
tool_position = obs.gripper_pose[:3]
tool_orientation = agent.quaternion_from_euler(np.array([[0.0, -1.0, 0.0]]))[0]
cap = cv2.VideoCapture(0)
detector = handDetector()
max_px = 720
pTime = 0
cTime = 0
while True:
    # time.sleep(0.05)

    success, img = cap.read()

    img = detector.findHands(img)
    lmlist = detector.findPosition(img)

    if len(lmlist) != 0:
        print("x : " + str(lmlist[4][1] / max_px))
        print("y : " + str(lmlist[4][2] / max_px))
        print("z : " + str(lmlist[0][3]))
        # dx = ((lmlist[4][1] / max_px) - (lmlist[8][1]/max_px))^2
        # dy = ((lmlist[4][1] / max_px) - (lmlist[8][1] / max_px)) ^ 2
        # open = (dx + dy)^(1/2)
        print("open' : " + str(open))
        tool_position[0] = (lmlist[0][1] / max_px)*(agent.max_dims[0]-agent.max_dims[0])+agent.min_dims[0]
        tool_position[2] = ((max_px - lmlist[0][2]) / max_px)*(agent.max_dims[2]-agent.max_dims[2])+agent.min_dims[2]
        print(tool_position)
        act = np.concatenate([tool_position, tool_orientation, [1.0]], axis=-1)
        # tcp_command = "movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0)\n" % (
        # tool_position[0], tool_position[1], tool_position[2], tool_orientation[0], tool_orientation[1],
        # tool_orientation[2], tool_acc, tool_vel)
        # s.send(str.encode(tcp_command))
        #rtde_c.moveL(tool_position + tool_orientation, tool_acc, tool_vel)
        print(act)
        print("step")
        obs, reward, terminate = task.step(act)
        print("step done")
        for k in range(15):
            success, img = cap.read()
            img = detector.findHands(img)
            lmlist = detector.findPosition(img)
            #cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            #cv2.imshow("Image", img)
            cv2.waitKey(1)
        # data = s.recv(1024)
        # print(data)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    #cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    #cv2.imshow("Image", img)
    cv2.waitKey(1)"""

env.shutdown()

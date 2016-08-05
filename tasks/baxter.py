import numpy as np
import cv2

import rospy
import baxter_interface
from std_msgs.msg import UInt16
from sensor_msgs.msg import CompressedImage
from baxter_interface import CHECK_VERSION

from utils.cooking_action import CookingActionModel


class CookingTask(object):
    def __init__(self, hyperparams):
        self.hyperparams = hyperparams
        self.state_len = hyperparams['state_len']
        self.joint_vels = {'up': hyperparams['joint_vel'],
                           'down': -hyperparams['joint_vel']}

        rospy.init_node('baxter_cooks', anonymous=True)
        self.setup_sim()
        # self.joints = ['e0', 'e1', 's0', 's1', 'w0', 'w1', 'w2']
        # self.actions = []
        # for joint in self.joints:
        #     for arm in ['left', 'right']:
        #         for direction in ['up', 'down']:
        #             self.actions.append((arm, joint, direction))
        self.actions = []
        for joint in self.right_joint_names:
            print('Joint: ' + joint)
            for direction in ['up', 'down']:
                self.actions.append((joint, direction))

        print('Num possible actions: %d' % len(self.actions))
        self.screen_size = (224, 224)
        self.state_shape = self.screen_size + (self.state_len,)

        rospy.Subscriber('/cameras/head_camera/image/compressed',
                         CompressedImage,
                         self.recv_image, queue_size=1)
        self.action_model = CookingActionModel()
        self.target_verb = 'cut'

    def recv_image(self, ros_data):
        np_array = np.fromstring(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        blue, green, red = image_np.T
        image_np = np.array([red, green, blue]).transpose()
        self.image = image_np

    def get_features(self, prev_image, image):
        return self.action_model.get_feats(prev_image, image)

    def get_image(self):
        return self.image

    def setup_sim(self):
        self.pub_rate = rospy.Publisher('robot/joint_state_publish_rate',
                                        UInt16, queue_size=10)
        self.left_arm = baxter_interface.limb.Limb("left")
        self.right_arm = baxter_interface.limb.Limb("right")
        self.left_joint_names = self.left_arm.joint_names()
        self.right_joint_names = self.right_arm.joint_names()
        # self.rate = 500.0  # Hz
        self.rate = 24
        print("Getting robot state... ")
        self.rs = baxter_interface.RobotEnable(CHECK_VERSION)
        self.init_state = self.rs.state().enabled
        print("Enabling robot... ")
        self.rs.enable()
        self.pub_rate.publish(self.rate)

    def reset_control_modes(self):
        rate = rospy.Rate(self._rate)
        for _ in range(100):
            if rospy.is_shutdown():
                return False
            self.left_arm.exit_control_mode()
            self.right_arm.exit_control_mode()
            self.pub_rate.publish(100)  # 100Hz default joint state rate
            rate.sleep()
        return True

    def clean_shutdown(self):
        print("\nExiting control...")
        self.reset_control_modes()
        self.set_neutral()
        if not self._init_state:
            print("Disabling robot...")
            self.rs.disable()
        return True

    def set_neutral(self):
        """
        Sets both arms back into a neutral pose.
        """
        print("Moving to neutral pose...")
        self.left_arm.move_to_neutral()
        self.right_arm.move_to_neutral()

    def preprocess_screen(self, screen_rgb):
        screen = np.dot(
            screen_rgb, np.array([.299, .587, .114])).astype(np.uint8)
        screen.resize(self.screen_size)
        # screen = screen / 255.0
        return np.array(screen)

    def preprocess_screen_rgb(self, screen_rgb):
        screen = np.copy(screen_rgb)
        screen.resize(self.screen_size + (3,))
        # screen = screen / 255.0
        return np.array(screen)

    def get_state_shape(self):
        return self.state_shape

    def get_actions(self):
        return list(range(len(self.actions)))

    def get_screen(self):
        return self.image

    def start_episode(self):
        self.set_neutral()
        self.episode_reward = 0
        self.step_num = 0
        self.recent_feats = []
        pre = self.preprocess_screen_rgb(self.get_screen())
        self.prev_pre = pre
        # self.recent_feats.append(self.get_features(pre, pre))
        self.states = []
        for _ in range(self.state_len):
            self.states.append(self.preprocess_screen(
                self.get_screen()))

    def get_episode_reward(self):
        return self.episode_reward

    def get_state(self):
        curr_state = np.stack(self.states, axis=2)
        return curr_state

    def compute_reward(self):
        if self.step_num == self.hyperparams['steps_per_episode'] - 1:
            num_frames = self.hyperparams['num_recent_feats']
            feats = np.stack(self.recent_feats[-num_frames:], axis=0)
            print(feats.shape)
            preds = self.action_model.get_preds_from_feats(feats)
            return preds[self.target_verb]
        else:
            return 0

    def act(self, action):
        right_joint_vels = {}
        for joint_name in self.right_joint_names:
            if joint_name == action[0]:
                right_joint_vels[joint_name] = self.joint_vels[action[1]]
            else:
                right_joint_vels[joint_name] = 0
        # left_joint_vels = {}
        # for joint_name in self.left_joint_names:
        #     if joint_name == action[0]:
        #         left_joint_vels[joint_name] = self.joint_vels[action[1]]
        #     else:
        #         left_joint_vels[joint_name] = 0
        self.right_arm.set_joint_velocities(right_joint_vels)
        # self.left_arm.set_joint_velocities(left_joint_vels)

    def perform_action(self, action_dist):
        action = self.actions[np.argmax(action_dist)]
        for _ in range(self.hyperparams['frame_skip']):
            self.act(action)
        pre = self.preprocess_screen_rgb(self.get_screen())
        self.recent_feats.append(self.get_features(self.prev_pre, pre))
        self.prev_pre = pre
        self.curr_reward = self.compute_reward()
        self.episode_reward += self.curr_reward
        screen = self.preprocess_screen(self.get_screen())
        self.states = self.states[:self.state_len - 1]
        self.states.insert(0, screen)
        self.step_num += 1

    def start_eval_mode(self):
        # self.init_ale(display=self.show_screen)
        pass

    def end_eval_mode(self):
        # self.init_ale(display=False)
        pass

    def get_reward(self):
        return self.curr_reward

    def episode_is_over(self):
        return self.step_num == self.hyperparams['steps_per_episode']

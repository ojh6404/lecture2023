import copy
import numpy as np
from rospkg import RosPack
from gym import utils, spaces
from gym.envs.mujoco.mujoco_env import MujocoEnv
import ramiel_utils


class RamielEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, test=False, ros=False, max_step=None):
        self.is_params_set = False
        self.test = test
        self.ros = ros
        self.max_step = max_step

        if self.test and self.ros:
            import rospy
            from std_msgs.msg import Float32MultiArray
            rospy.init_node("ramiel_env")
            self.debug_msg = Float32MultiArray()
            self.debug_pub = rospy.Publisher("ramiel_env/debug", Float32MultiArray, queue_size=10)

        self.rospack = RosPack()
        model_path = self.rospack.get_path("mujoco_tutorials") + "/models/ramiel_robot.xml"
        MujocoEnv.__init__(self, model_path, 5)

        utils.EzPickle.__init__(self)

    def set_param(self):
        # robot joint/pose id
        self.robot_pose_indices = self.model.jnt_qposadr[1:]  # [worldpos(3), worldquat(4), jointpos(3)]
        self.robot_vel_indices = self.model.jnt_dofadr[1:]  # [worldve(3), worldrotvel(3), jointvel(3)]
        self.pole_geom_id = self.model.geom_name2id("leg_mesh")
        self.support_geom_indices = [self.model.geom_name2id("supportA"), self.model.geom_name2id("supportB"), self.model.geom_name2id("supportC")]

        # sensor id
        self.touch_sensor_id = self.model.sensor_name2id("contact_sensor")
        self.accelerometer_id = self.model.sensor_name2id("accelerometer")
        self.gyro_id = self.model.sensor_name2id("gyro")
        self.framequat_id = self.model.sensor_name2id("framequat")
        self.velocimeter_id = self.model.sensor_name2id("velocimeter")
        self.framepos_id = self.model.sensor_name2id("framepos")

        self.ctrl_min = [-50, -50, -320]
        self.ctrl_max = [50, 50, 90]
        self.n_prev = 6
        self.qpos_rand = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02], dtype=np.float32) # quaternion (4) + joint angles (3) = (7)
        self.const_qpos_rand = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02], dtype=np.float32) # quaternion (4) + joint angles (3) = (7)
        self.qvel_rand = np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.1], dtype=np.float32) # velocity of quaternion (3) + joint velocities (3) = (6)
        self.bvel_rand = np.array([0.1, 0.1, 0.5], dtype=np.float32) # body velocity
        self.force_rand = np.array([3.0, 3.0, 3.0], dtype=np.float32)
        self.const_force_rand = np.array([3.0, 3.0, 3.0], dtype=np.float32)
        self.torque_rand = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        self.const_torque_rand = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        self.action_rand = np.array([0.05, 0.05, 0.05], dtype=np.float32)
        self.const_action_rand = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        self.max_episode = 10000

        if self.test:
            self.default_step_rate = 0.5

        # variable for rl
        self.const_ext_qpos = np.zeros(7)
        self.const_ext_force = np.zeros(3)
        self.const_ext_torque = np.zeros(3)
        self.const_ext_action = np.zeros(3)
        self.current_qpos = None
        self.current_qvel = None # not used
        self.current_bvel = None
        self.prev_qpos = None
        self.prev_qvel = None # not used
        self.prev_bvel = None
        self.prev_action = None
        self.episode_cnt = 0
        self.step_cnt = 0

        # print("action space: {}".format(self.action_space))

    def step(self, action): # action = [tau_x, tau_y, tau_slide] (3)
        if not self.is_params_set:
            self.set_param()
            self.is_params_set = True

        if self.max_step:
            step_rate = float(self.step_cnt)/self.max_step
        elif self.test:
            step_rate = self.default_step_rate

        # quaternion (4) + joint angles of slide/roll/pitch (3) = (7)
        if self.current_qpos is None:
            self.current_qpos = self.sim.data.qpos.flat[3:]

        # velocity of quaternion (3) + joint velocity of roll/pitch/slide (3) = (6)
        if self.current_qvel is None:
            self.current_qvel = self.sim.data.qvel.flat[3:]

        if self.current_bvel is None: # body xyz linear velocity (3)
            self.current_bvel = self.sim.data.qvel.flat[:3]

        if self.prev_action is None:
            self.prev_action = [copy.deepcopy(action) for i in range(self.n_prev)]

        if self.prev_qpos is None:
            self.prev_qpos = [copy.deepcopy(self.current_qpos) for i in range(self.n_prev)]

        if self.prev_qvel is None:
            self.prev_qvel = [copy.deepcopy(self.current_qvel) for i in range(self.n_prev)]

        if self.prev_bvel is None:
            self.prev_bvel = [copy.deepcopy(self.current_bvel) for i in range(self.n_prev)]

        pose = self.prev_qpos[-1][4:] # joint angle (3)
        vel = self.prev_qvel[-1][3:] # joint velocity (3)
        jacobian = ramiel_utils.pose2jacobian(pose[0], pose[1], pose[2])
        # add random noise
        action_rate = 1.0 + self.action_rand*step_rate*np.random.randn(3) + self.const_ext_action
        action_converted = [(cmin+(rate*a+1.0)*(cmax-cmin)/2.0) for a, cmin, cmax, rate in zip(action, self.ctrl_min, self.ctrl_max, action_rate)] # joint torque (3)
        tension_ref_mat = ramiel_utils.compute_tension(jacobian, action_converted)
        tension_ref = [t  for t in tension_ref_mat]
        self.sim.data.qfrc_applied[:3] = self.const_ext_force + self.force_rand*step_rate*np.random.randn(3) # body linear force [N]
        self.sim.data.qfrc_applied[-3:] = self.const_ext_torque + self.torque_rand*step_rate*np.random.randn(3) # joint torque force [Nm]

        # np.set_printoptions(precision=3)
        # np.set_printoptions(suppress=True)

        # do simulation
        self.do_simulation(tension_ref, self.frame_skip)
        # print([self.sim.data.sensordata[self.model.sensor_adr[self.model.sensor_name2id(name)]] for name in ["dA_top", "dB_top", "dC_top", "dA_bottom", "dB_bottom", "dC_bottom"]])

        # next state without noise to calculate reward
        pose = self.sim.data.qpos[self.robot_pose_indices] # joint angle
        vel = self.sim.data.qvel[self.robot_vel_indices] # joint velocity
        quat = self.sim.data.qpos[[4, 5, 6, 3]] # [x, y, z, w]
        pole_quat = self.sim.data.body_xquat[self.model.nbody-2][[1, 2, 3, 0]]

        # reward definition
        jump_reward = 0.0
        govel_reward = 0.0 # the difference between the current base vel and target base vel
        rotate_reward = 0.0 # do not rotate the base in the direction of yaw
        horizontal_reward = 0.0 # do not slant the pole and base
        ctrl_reward = 0.0 # restraint for joint action (torque)
        contact_reward = 0.0 # restraint for contact between ground and pose/support
        survive_reward = 0.0
        range_reward = 0.0 # restraint for joint range limitation

        if self.sim.data.qpos[2] > 1.2:
            jump_reward = -1.0*step_rate
        else:
            jump_reward = 10.0*min(0.8, self.sim.data.qpos[2])**2
            # jump_reward = 1.0/min(max(0.1, step_rate), 0.5)*self.sim.data.qpos[2]**2
        govel_reward = -1.0*step_rate*np.square(self.sim.data.qvel[[0, 1]]).sum()
        rotate_reward = -10.0*step_rate*np.square(self.sim.data.qvel[3:6]).sum()
        horizontal_reward = -3.0*(1.0-ramiel_utils.horizontal_eval(quat))-1.0*(1.0-ramiel_utils.horizontal_eval(pole_quat))
        ctrl_reward = -0.3*step_rate*np.square(np.array([10.0, 10.0, 1.0])*action).sum() # very important
        if any([(self.sim.data.contact[nc].geom2 in [self.pole_geom_id]) for nc in range(self.sim.data.ncon)]):
            contact_reward += -0.3
        if any([(self.sim.data.contact[nc].geom2 in self.support_geom_indices) for nc in range(self.sim.data.ncon)]):
            contact_reward += -0.3
        if self.sim.data.site_xpos[self.model.site_name2id("contact_sensor")][2] < 0.0:
            contact_reward += -10000.0*step_rate*(self.sim.data.site_xpos[self.model.site_name2id("contact_sensor")][2])**2
        # else:
        #     contact_reward += 30.0*step_rate*(min(0.2, self.sim.data.site_xpos[self.model.site_name2id("contact_sensor")][2]))**2
        survive_reward = 0.1
        range_reward = ramiel_utils.range_reward(pose[0], pose[1], pose[2])

        reward = jump_reward + govel_reward + rotate_reward + horizontal_reward + ctrl_reward + contact_reward + survive_reward + range_reward

        if self.test and self.ros:
            self.debug_msg.data = np.concatenate([np.array(action_converted), pose, vel, self.sim.data.qvel[:6]])
            self.debug_pub.publish(self.debug_msg)

        self.episode_cnt += 1
        self.step_cnt += 1
        notdone = ramiel_utils.check_range(pose[0], pose[1], pose[2])
        notdone &= ramiel_utils.horizontal_eval(pole_quat) > 0.5
        notdone &= self.episode_cnt < self.max_episode
        if self.step_cnt == 1:
            done = False
        else:
            done = not notdone

        self.current_qpos = self.sim.data.qpos.flat[3:] + self.qpos_rand*step_rate*np.random.randn(7) + self.const_ext_qpos
        self.current_qvel = self.sim.data.qvel.flat[3:] + self.qvel_rand*step_rate*np.random.randn(6)
        self.current_bvel = self.sim.data.qvel.flat[:3] + self.bvel_rand*step_rate*np.random.randn(3)
        self.prev_qpos.append(copy.deepcopy(self.current_qpos))
        self.prev_qvel.append(copy.deepcopy(self.current_qvel))
        self.prev_bvel.append(copy.deepcopy(self.current_bvel))
        if len(self.prev_qpos) > self.n_prev:
            del self.prev_qpos[0]
        if len(self.prev_qvel) > self.n_prev:
            del self.prev_qvel[0]
        if len(self.prev_bvel) > self.n_prev:
            del self.prev_bvel[0]
        obs = self._get_obs()
        self.prev_action.append(copy.deepcopy(action))
        if len(self.prev_action) > self.n_prev:
            del self.prev_action[0]
        if done:
            self.episode_cnt = 0
            self.current_qpos = None
            self.current_qvel = None
            self.prev_action = None
            self.prev_qpos = None
            self.prev_qvel = None
            self.prev_bvel = None
            self.const_ext_qpos = self.const_qpos_rand*step_rate*np.random.randn(7)
            self.const_ext_force = self.const_force_rand*step_rate*np.random.randn(3)
            self.const_ext_torque = self.const_torque_rand*step_rate*np.random.randn(3)
            self.const_ext_action = self.const_action_rand*step_rate*np.random.randn(3)
        return (
            obs,
            reward,
            done,
            dict(
                jump_reward = jump_reward,
                govel_reward = govel_reward,
                rotate_reward = rotate_reward,
                horizontal_reward = horizontal_reward,
                ctrl_reward = ctrl_reward,
                contact_reward = contact_reward,
                survive_reward = survive_reward,
                range_reward = range_reward,
                ),
            )

    def _get_obs(self):
        if self.max_step:
            step_rate = float(self.step_cnt)/self.max_step
        elif self.test:
            step_rate = self.default_step_rate
        # accel = self.sim.data.sensordata[self.accelerometer_id:self.accelerometer_id+3]
        # accel[2] -= 9.8
        return np.concatenate(
            [
                np.concatenate(self.prev_qpos), # prev base quat + joint angles
                np.concatenate(self.prev_qvel), # prev base quat vel + joint vels
                np.concatenate(self.prev_bvel), # prev base linear vel
                np.concatenate(self.prev_action), # prev action
            ]
        )

    def _set_action_space(self):
        # bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        # low, high = bounds.T
        low = np.asarray([-1.0, -1.0, -1.0], dtype=np.float32)
        high = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def reset_model(self):
        slide_qpos_id = self.model.jnt_qposadr[self.model.joint_name2id("slide")]
        roll_qpos_id = self.model.jnt_qposadr[self.model.joint_name2id("roll")]
        pitch_qpos_id = self.model.jnt_qposadr[self.model.joint_name2id("pitch")]

        if self.max_step:
            step_rate = float(self.step_cnt)/self.max_step
        elif self.test:
            step_rate = self.default_step_rate

        qpos = self.init_qpos
        qpos[2] = 0.19
        qpos[slide_qpos_id] = 0.874
        qpos[roll_qpos_id] = 0.03*step_rate*np.random.randn(1)
        qpos[pitch_qpos_id] = 0.03*step_rate*np.random.randn(1)
        qvel = self.init_qvel
        self.set_state(qpos, qvel)

        if (self.prev_qpos is None) and (self.prev_action is None):
            self.current_qpos = self.sim.data.qpos.flat[3:]
            self.current_qvel = self.sim.data.qvel.flat[3:]
            self.current_bvel = self.sim.data.qvel.flat[:3]
            self.prev_action = [np.zeros(3) for i in range(self.n_prev)]
            self.prev_qpos = [self.current_qpos + self.qpos_rand*step_rate*np.random.randn(7) for i in range(self.n_prev)]
            self.prev_qvel = [self.current_qvel + self.qvel_rand*step_rate*np.abs(self.sim.data.qvel.flat[3:])*np.random.randn(6) for i in range(self.n_prev)]
            self.prev_bvel = [self.current_bvel + self.bvel_rand*step_rate*np.abs(self.sim.data.qvel.flat[:3])*np.random.randn(3) for i in range(self.n_prev)]

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 1.0


#!/usr/bin/env python3
import copy
import numpy as np
from gym import utils, spaces
from gym.envs.mujoco.mujoco_env import MujocoEnv


class KHRMimicEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, test=False, max_step=None):
        self.is_params_set = False
        self.test = test
        self.max_step = max_step
        frame_skip = 5

        motion_data = np.load("07_08.npz", allow_pickle=True)

        self.retarget_frames = motion_data["retarget_frames"]
        self.ref_joint_pos = motion_data["ref_joint_pos"]
        self.robot_joint_indices = motion_data["robot_joint_indices"]
        self.non_fixed_joint_indices = motion_data["non_fixed_joint_indices"]
        self.frame_duration = motion_data["frame_duration"]

        self.num_frames = self.retarget_frames.shape[0]

        self.ref_base_pos = self.retarget_frames[:, 0:3]  # x, y, z [m]
        self.ref_base_quat = self.retarget_frames[:, 3:7]  # x, y, z, w
        self.ref_joint_pos = self.retarget_frames[:, 7:]  # joint angles [rad]

        print("num_frames: ", self.num_frames)
        print("frame_duration: ", self.frame_duration)
        print("ref_base_pos: ", self.ref_base_pos.shape)
        print("ref_base_quat: ", self.ref_base_quat.shape)
        print("ref_joint_pos: ", self.ref_joint_pos.shape)
        print("robot_joint_indices: ", self.robot_joint_indices)
        print("non_fixed_joint_indices: ", self.non_fixed_joint_indices)

        # LARM LLEG RARM RLEG

        model_path = "/home/oh/ros/lecture_ws/src/agent-system/lecture2023/student_projects/khr_mimic/models/KHR/KHR.xml"
        MujocoEnv.__init__(self, model_path, frame_skip=frame_skip)
        utils.EzPickle.__init__(self)

    def set_param(self):
        # get joint/pose id
        self.jnt_pos_indices = self.model.jnt_qposadr[1:]  # joint pos indices
        self.jnt_vel_indices = self.model.jnt_dofadr[1:]  # joint vel indices

        self.n_joints = len(self.jnt_pos_indices)

        # get geom id
        self.lleg_geom_id = self.model.geom_name2id("lleg_link4_mesh")
        self.rleg_geom_id = self.model.geom_name2id("rleg_link4_mesh")

        # get sensor id
        # self.touch_sensor_id = self.model.sensor_name2id("contact_sensor")
        self.accelerometer_id = self.model.sensor_name2id("accelerometer")
        self.gyro_id = self.model.sensor_name2id("gyro")
        self.framequat_id = self.model.sensor_name2id("framequat")
        self.velocimeter_id = self.model.sensor_name2id("velocimeter")
        self.framepos_id = self.model.sensor_name2id("framepos")

        self.torque_max = 2.5  # [Nm]
        self.kp = 10.0
        self.kd = 0.5
        self.ctrl_min = np.ones(self.n_joints, dtype=np.float32) * -self.torque_max
        self.ctrl_max = np.ones(self.n_joints, dtype=np.float32) * self.torque_max

        self.n_prev = 6
        self.qpos_rand = np.array(
            [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02] * 3, dtype=np.float32
        )  # quaternion (4) + joint angles (17) = (21)
        self.const_qpos_rand = np.array(
            [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02] * 3, dtype=np.float32
        )  # quaternion (4) + joint angles (17) = (21)
        self.qvel_rand = np.array(
            [0.1, 0.1, 0.1, 0.3, 0.3, 0.1] * 3 + [0.1, 0.1], dtype=np.float32
        )  # velocity of quaternion (3) + joint velocities (17) = (20)
        self.bvel_rand = np.array([0.1, 0.1, 0.5], dtype=np.float32)  # body velocity
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
        self.const_ext_qpos = np.zeros(21)
        self.const_ext_force = np.zeros(3)
        self.const_ext_torque = np.zeros(3)
        self.const_ext_action = np.zeros(3)
        self.current_qpos = None
        self.current_qvel = None  # not used
        self.current_bvel = None
        self.prev_qpos = None
        self.prev_qvel = None  # not used
        self.prev_bvel = None
        self.prev_action = None
        self.episode_cnt = 0
        self.step_cnt = 0

        self.base_pos = None
        self.base_quat = None
        self.base_vel = None
        self.base_angvel = None
        self.jnt_pos = None
        self.jnt_vel = None
        self.jnt_torque = None

        print("action space: {}".format(self.action_space.shape))

    def step(self, action):  # action : joint torque (17)
        if not self.is_params_set:
            self.set_param()
            self.is_params_set = True

        if self.max_step:
            step_rate = float(self.step_cnt) / self.max_step
        elif self.test:
            step_rate = self.default_step_rate

        if self.current_qpos is None:
            self.current_qpos = self.sim.data.qpos.flat[3:]

        if self.base_pos is None:
            self.base_pos = self.sim.data.qpos.flat[3:]

        # velocity of quaternion (3) + joint velocity of roll/pitch/slide (3) = (6)
        if self.current_qvel is None:
            self.current_qvel = self.sim.data.qvel.flat[3:]

        if self.current_bvel is None:  # body xyz linear velocity (3)
            self.current_bvel = self.sim.data.qvel.flat[:3]

        if self.prev_action is None:
            self.prev_action = [copy.deepcopy(action) for i in range(self.n_prev)]

        if self.prev_qpos is None:
            self.prev_qpos = [
                copy.deepcopy(self.current_qpos) for i in range(self.n_prev)
            ]

        if self.prev_qvel is None:
            self.prev_qvel = [
                copy.deepcopy(self.current_qvel) for i in range(self.n_prev)
            ]

        if self.prev_bvel is None:
            self.prev_bvel = [
                copy.deepcopy(self.current_bvel) for i in range(self.n_prev)
            ]

        pose = self.prev_qpos[-1][4:]  # joint angle (3)
        vel = self.prev_qvel[-1][3:]  # joint velocity (3)
        # jacobian = ramiel_utils.pose2jacobian(pose[0], pose[1], pose[2])
        # add random noise
        action_rate = (
            1.0
            + self.action_rand * step_rate * np.random.randn(3)
            + self.const_ext_action
        )
        # action_converted = [
        #     (cmin + (rate * a + 1.0) * (cmax - cmin) / 2.0)
        #     for a, cmin, cmax, rate in zip(
        #         action, self.ctrl_min, self.ctrl_max, action_rate
        #     )
        # ]  # joint torque (3)
        self.sim.data.qfrc_applied[
            :3
        ] = self.const_ext_force + self.force_rand * step_rate * np.random.randn(
            3
        )  # body linear force [N]
        self.sim.data.qfrc_applied[
            -3:
        ] = self.const_ext_torque + self.torque_rand * step_rate * np.random.randn(
            3
        )  # joint torque force [Nm]

        # np.set_printoptions(precision=3)
        # np.set_printoptions(suppress=True)

        # do simulation
        # 0 : head_joint0, 1 : larm_joint0, 2 : larm_joint1 3 : larm_joint2
        # 4 : rarm_joint0, 5 : rarm_joint1, 6 : rarm_joint2
        # 7 : lleg_joint0, 8 : lleg_joint1, 9 : lleg_joint2, 10 : lleg_joint3, 11 : lleg_joint4
        # 12 : rleg_joint0, 13 : rleg_joint1, 14 : rleg_joint2, 15 : rleg_joint3, 16 : rleg_joint4
        torque = np.zeros(17, dtype=np.float32)
        self.do_simulation(torque, self.frame_skip)
        # print([self.sim.data.sensordata[self.model.sensor_adr[self.model.sensor_name2id(name)]] for name in ["dA_top", "dB_top", "dC_top", "dA_bottom", "dB_bottom", "dC_bottom"]])

        # next state without noise to calculate reward
        jnt_pos = self.sim.data.qpos[self.jnt_pos_indices]  # joint angle
        jnt_vel = self.sim.data.qvel[self.jnt_vel_indices]  # joint velocity
        base_quat = self.sim.data.qpos[[4, 5, 6, 3]]  # [x, y, z, w]
        # pole_quat = self.sim.data.body_xquat[self.model.nbody - 2][[1, 2, 3, 0]]

        # reward definition
        jump_reward = 0.0
        govel_reward = (
            0.0  # the difference between the current base vel and target base vel
        )
        rotate_reward = 0.0  # do not rotate the base in the direction of yaw
        horizontal_reward = 0.0  # do not slant the pole and base
        ctrl_reward = 0.0  # restraint for joint action (torque)
        contact_reward = 0.0  # restraint for contact between ground and pose/support
        survive_reward = 0.0
        range_reward = 0.0  # restraint for joint range limitation

        # if self.sim.data.qpos[2] > 1.2:
        #     jump_reward = -1.0 * step_rate
        # else:
        #     jump_reward = 10.0 * min(0.8, self.sim.data.qpos[2]) ** 2
        #     # jump_reward = 1.0/min(max(0.1, step_rate), 0.5)*self.sim.data.qpos[2]**2
        # govel_reward = -1.0 * step_rate * np.square(self.sim.data.qvel[[0, 1]]).sum()
        # rotate_reward = -10.0 * step_rate * np.square(self.sim.data.qvel[3:6]).sum()
        # ctrl_reward = (
        #     -0.3 * step_rate * np.square(np.array([10.0, 10.0, 1.0]) * action).sum()
        # )  # very important
        # if any(
        #     [
        #         (self.sim.data.contact[nc].geom2 in [self.pole_geom_id])
        #         for nc in range(self.sim.data.ncon)
        #     ]
        # ):
        #     contact_reward += -0.3
        # if any(
        #     [
        #         (self.sim.data.contact[nc].geom2 in self.support_geom_indices)
        #         for nc in range(self.sim.data.ncon)
        #     ]
        # ):
        #     contact_reward += -0.3
        # if self.sim.data.site_xpos[self.model.site_name2id("contact_sensor")][2] < 0.0:
        #     contact_reward += (
        #         -10000.0
        #         * step_rate
        #         * (
        #             self.sim.data.site_xpos[self.model.site_name2id("contact_sensor")][
        #                 2
        #             ]
        #         )
        #         ** 2
        #     )

        # else:
        #     contact_reward += 30.0*step_rate*(min(0.2, self.sim.data.site_xpos[self.model.site_name2id("contact_sensor")][2]))**2
        survive_reward = 0.1

        reward = (
            # jump_reward
            # + govel_reward
            # + rotate_reward
            # + ctrl_reward
            # + contact_reward
            survive_reward
            # + range_reward
        )

        # if self.test and self.ros:
        #     self.debug_msg.data = np.concatenate(
        #         [np.array(action_converted), pose, vel, self.sim.data.qvel[:6]]
        #     )
        #     self.debug_pub.publish(self.debug_msg)

        self.episode_cnt += 1
        self.step_cnt += 1
        # notdone = ramiel_utils.check_range(pose[0], pose[1], pose[2])
        # notdone &= ramiel_utils.horizontal_eval(pole_quat) > 0.5
        notdone = self.episode_cnt < self.max_episode
        if self.step_cnt == 1:
            done = False
        else:
            done = not notdone

        self.current_qpos = (
            self.sim.data.qpos.flat[3:]
            + self.qpos_rand * step_rate * np.random.randn(21)
            + self.const_ext_qpos
        )
        self.current_qvel = self.sim.data.qvel.flat[
            3:
        ] + self.qvel_rand * step_rate * np.random.randn(20)
        self.current_bvel = self.sim.data.qvel.flat[
            :3
        ] + self.bvel_rand * step_rate * np.random.randn(3)
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
            self.const_ext_qpos = self.const_qpos_rand * step_rate * np.random.randn(7)
            self.const_ext_force = (
                self.const_force_rand * step_rate * np.random.randn(3)
            )
            self.const_ext_torque = (
                self.const_torque_rand * step_rate * np.random.randn(3)
            )
            self.const_ext_action = (
                self.const_action_rand * step_rate * np.random.randn(3)
            )
        return (
            obs,
            reward,
            done,
            dict(
                jump_reward=jump_reward,
                govel_reward=govel_reward,
                rotate_reward=rotate_reward,
                horizontal_reward=horizontal_reward,
                ctrl_reward=ctrl_reward,
                contact_reward=contact_reward,
                survive_reward=survive_reward,
                range_reward=range_reward,
            ),
        )

    def _get_obs(self):
        if self.max_step:
            step_rate = float(self.step_cnt) / self.max_step
        elif self.test:
            step_rate = self.default_step_rate
        # accel = self.sim.data.sensordata[self.accelerometer_id:self.accelerometer_id+3]
        # accel[2] -= 9.8
        return np.concatenate(
            [
                np.concatenate(self.prev_qpos),  # prev base quat + joint angles
                np.concatenate(self.prev_qvel),  # prev base quat vel + joint vels
                np.concatenate(self.prev_bvel),  # prev base linear vel
                np.concatenate(self.prev_action),  # prev action
            ]
        )

    def _set_action_space(self):
        # bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        # low, high = bounds.T
        low = np.ones(17, dtype=np.float32) * -1.0
        high = np.ones(17, dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def reset_model(self):
        slide_qpos_id = self.model.jnt_qposadr[
            self.model.joint_name2id("KHR/rleg_joint0")
        ]
        roll_qpos_id = self.model.jnt_qposadr[
            self.model.joint_name2id("KHR/rleg_joint2")
        ]
        pitch_qpos_id = self.model.jnt_qposadr[
            self.model.joint_name2id("KHR/rleg_joint1")
        ]

        print("head joint id")
        print(self.model.joint_name2id("KHR/head_joint0"))

        if self.max_step:
            step_rate = float(self.step_cnt) / self.max_step
        elif self.test:
            step_rate = self.default_step_rate

        qpos = self.init_qpos
        print("qpos", qpos.shape)
        qpos[2] = 0.3
        qpos[slide_qpos_id] = 0.874
        qpos[roll_qpos_id] = 0.03 * step_rate * np.random.randn(1)
        qpos[pitch_qpos_id] = 0.03 * step_rate * np.random.randn(1)
        qvel = self.init_qvel
        qpos[7:] = 0.0
        qvel[:] = 0.0
        self.set_state(qpos, qvel)

        if (self.prev_qpos is None) and (self.prev_action is None):
            self.current_qpos = self.sim.data.qpos.flat[3:]
            self.current_qvel = self.sim.data.qvel.flat[3:]
            self.current_bvel = self.sim.data.qvel.flat[:3]
            self.prev_action = [np.zeros(3) for i in range(self.n_prev)]
            self.prev_qpos = [
                self.current_qpos + self.qpos_rand * step_rate * np.random.randn(7)
                for i in range(self.n_prev)
            ]
            self.prev_qvel = [
                self.current_qvel
                + self.qvel_rand
                * step_rate
                * np.abs(self.sim.data.qvel.flat[3:])
                * np.random.randn(6)
                for i in range(self.n_prev)
            ]
            self.prev_bvel = [
                self.current_bvel
                + self.bvel_rand
                * step_rate
                * np.abs(self.sim.data.qvel.flat[:3])
                * np.random.randn(3)
                for i in range(self.n_prev)
            ]

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 1.0

        # retarget_frames=retarget_frames,
        # ref_joint_pos=ref_joint_pos,
        # robot_joint_indices=robot_joint_indices,
        # non_fixed_joint_indices=get_non_fixed_joint_indices(robot),
        # frame_duration=bvh_cfg.FRAME_DURATION,


if __name__ == "__main__":
    env = KHRMimicEnv(test=True)
    env.reset()

    # for i in range(1000):
    #     env.step(np.zeros(17))
    #     env.render()

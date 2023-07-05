#!/usr/bin/env python3

import os
import copy
import numpy as np
from gym import utils, spaces
from gym.envs.mujoco.mujoco_env import MujocoEnv
from mujoco_py.generated import const
from utils import *

"""
qpos
0 : base_x, 1 : base_y, 2 : base_z
3 : base_quat_w, 4 : base_quat_x, 5 : base_quat_y, 6 : base_quat_z
7 : rleg_joint0, 8 : rleg_joint1, 9 : rleg_joint2, 10 : rleg_joint3, 11 : rleg_joint4
12 : lleg_joint0, 13 : lleg_joint1, 14 : lleg_joint2, 15 : lleg_joint3, 16 : lleg_joint4
17 : rarm_joint0, 18 : rarm_joint1, 19 : rarm_joint2
20 : larm_joint0, 21 : larm_joint1, 22 : larm_joint2
23 : head_joint0

jnt_idx = qpos_idx - 7

"""


class KHRMimicEnv(MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        test=False,
        max_step=20000000,
        early_termination=False,
        control_type="torque",
    ):
        self.is_params_set = False
        self.test = test
        self.max_step = max_step  # 20000000
        self.early_termination = early_termination
        self.control_type = control_type
        frame_skip = 10  # for 50 Hz control

        root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        motion_data_path = os.path.join(root_dir, "data", "blended_processed_07_08.npz")
        model_path = os.path.join(root_dir, "models", "KHR", "KHR.xml")
        self.load_motion_data(motion_data_path)
        MujocoEnv.__init__(self, model_path, frame_skip=frame_skip)
        utils.EzPickle.__init__(self)
        print("sim dt", self.dt)

    def load_motion_data(self, motion_data_path):
        motion_data = np.load(
            motion_data_path,
            allow_pickle=True,
        )
        """
        motion data order
        RLEG LLEG RARM LARM
        """
        # NOTE original cycle frames : 150, 120 Hz -> adjust to 50 HZ
        original_cycle_frames = 150
        original_frame_duration = 1.0 / 120.0  # 120 hz
        dt = 1.0 / 50.0  # 50 hz
        self.motion_cycle_frames = int(
            original_cycle_frames * (original_frame_duration / dt)
        )  # 62

        self.ref_base_pos = motion_data["ref_base_pos"][0 : self.motion_cycle_frames]
        self.ref_base_quat = motion_data["ref_base_quat"][0 : self.motion_cycle_frames]
        self.ref_jnt_pos = motion_data["ref_jnt_pos"][0 : self.motion_cycle_frames]
        self.ref_jnt_vel = motion_data["ref_jnt_vel"][0 : self.motion_cycle_frames]
        self.ref_base_lin_vel = motion_data["ref_base_lin_vel"][
            0 : self.motion_cycle_frames
        ]
        self.ref_base_ang_vel = motion_data["ref_base_ang_vel"][
            0 : self.motion_cycle_frames
        ]
        self.ref_ee_pos = motion_data["ref_ee_pos"][0 : self.motion_cycle_frames]
        self.frame_duration = motion_data["frame_duration"]
        self.num_frames = motion_data["num_frames"]  # not used

        self.motion_cycle_period = self.motion_cycle_frames * self.frame_duration

    def set_param(self):
        # get joint/pose id
        self.jnt_pos_indices = self.model.jnt_qposadr[1:]  # joint pos indices
        self.jnt_vel_indices = self.model.jnt_dofadr[1:]  # joint vel indices
        self.n_joints = len(self.jnt_pos_indices)
        self.n_control_joints = self.n_joints - 1  # NOTE without head joint

        # get geom id
        self.floor_geom_id = self.model.geom_name2id("floor")
        self.lleg_link4_geom_id = self.model.geom_name2id("lleg_link4_mesh")
        self.rleg_link4_geom_id = self.model.geom_name2id("rleg_link4_mesh")
        self.head_geom_id = self.model.geom_name2id("head_link0_mesh")

        self.terminal_contact_geom_ids = [
            self.model.geom_name2id(geom_names)
            for geom_names in self.model.geom_names
            if geom_names not in ["floor", "lleg_link4_mesh", "rleg_link4_mesh"]
        ]

        # get sensor id
        # self.touch_sensor_id = self.model.sensor_name2id("contact_sensor")
        # self.accelerometer_id = self.model.sensor_name2id("accelerometer")
        # self.gyro_id = self.model.sensor_name2id("gyro")
        # self.framequat_id = self.model.sensor_name2id("framequat")
        # self.velocimeter_id = self.model.sensor_name2id("velocimeter")
        # self.framepos_id = self.model.sensor_name2id("framepos")

        self.torque_max = 5.0  # [Nm]
        self.kp = 10.0  # TODO
        self.kd = 0.03  # TODO

        self.buffer = None
        self.buffer_size = 3

        # variable for rl
        self.max_episode = 1000  # 20 [s]
        self.episode_cnt = 0  # local
        self.step_cnt = 0  # global
        self.frame_cnt = 0
        self.time = 0  # [s]
        self.episode_reward = 0.0

        self.episode_mimic_jnt_pos_reward = 0.0
        self.episode_mimic_jnt_vel_reward = 0.0
        self.episode_mimic_ee_pos_reward = 0.0
        self.episode_mimic_base_pos_reward = 0.0
        self.episode_mimic_base_quat_reward = 0.0
        self.episode_mimic_base_lin_vel_reward = 0.0

        self.init_motion_data_frame = 0

    def set_buffer(self):
        action = self.action
        base_pos_z = self.sim.data.qpos[2, np.newaxis]
        base_quat = self.sim.data.qpos[3:7]
        jnt_pos = self.sim.data.qpos[
            7 : 7 + self.n_control_joints
        ]  # NOTE no head joint
        base_lin_vel = self.sim.data.qvel[:3]
        base_ang_vel = self.sim.data.qvel[3:6]
        if self.buffer is None:  # reset
            self.buffer = dict()
            self.buffer["action"] = [
                copy.deepcopy(action) for _ in range(self.buffer_size)
            ]
            self.buffer["base_pos_z"] = [
                copy.deepcopy(base_pos_z) for _ in range(self.buffer_size)
            ]
            self.buffer["base_quat"] = [
                copy.deepcopy(base_quat) for _ in range(self.buffer_size)
            ]
            self.buffer["jnt_pos"] = [
                copy.deepcopy(jnt_pos) for _ in range(self.buffer_size)
            ]
            self.buffer["base_lin_vel"] = [
                copy.deepcopy(base_lin_vel) for _ in range(self.buffer_size)
            ]
            self.buffer["base_ang_vel"] = [
                copy.deepcopy(base_ang_vel) for _ in range(self.buffer_size)
            ]
        else:  # update buffer
            self.buffer["action"].append(copy.deepcopy(action))
            self.buffer["base_pos_z"].append(copy.deepcopy(base_pos_z))
            self.buffer["base_quat"].append(copy.deepcopy(base_quat))
            self.buffer["jnt_pos"].append(copy.deepcopy(jnt_pos))
            self.buffer["base_lin_vel"].append(copy.deepcopy(base_lin_vel))
            self.buffer["base_ang_vel"].append(copy.deepcopy(base_ang_vel))
            self.buffer["action"].pop(0)
            self.buffer["base_pos_z"].pop(0)
            self.buffer["base_quat"].pop(0)
            self.buffer["jnt_pos"].pop(0)
            self.buffer["base_lin_vel"].pop(0)
            self.buffer["base_ang_vel"].pop(0)

    def floor_contact_check(self, geom_id):
        if any(
            [
                self.sim.data.contact[nc].geom2 == geom_id
                for nc in range(self.sim.data.ncon)
                if self.model.geom_bodyid[self.sim.data.contact[nc].geom1]
                == self.floor_geom_id
            ]
        ):
            return True
        else:
            return False

    def step(self, action):  # action : joint torque (17)
        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)

        # set params if none
        self.action = action[:].copy()  # without head joint
        if not self.is_params_set:
            self.set_param()
            self.set_buffer()
            self.is_params_set = True

        # compute time step
        self.local_frame_cnt = (
            self.init_motion_data_frame + self.frame_cnt
        )  # frame in motion data
        self.loop_cnt = (
            self.local_frame_cnt // self.motion_cycle_frames
        )  # frame loop count
        self.cycle_frame_cnt = (
            self.local_frame_cnt % self.motion_cycle_frames
        )  # frame in cycle
        self.time += self.dt

        # get reference motion for time step
        ref_base_pos = self.ref_base_pos[self.cycle_frame_cnt] + np.array(
            [self.ref_base_pos[-1, 0] * self.loop_cnt, 0.0, 0.0]
        )
        ref_base_quat = self.ref_base_quat[self.cycle_frame_cnt]
        ref_jnt_pos = self.ref_jnt_pos[self.cycle_frame_cnt]
        ref_jnt_vel = self.ref_jnt_vel[self.cycle_frame_cnt]
        ref_ee_pos = self.ref_ee_pos[self.cycle_frame_cnt]  # NOTE local
        ref_base_lin_vel = self.ref_base_lin_vel[self.cycle_frame_cnt]
        ref_base_ang_vel = self.ref_base_ang_vel[self.cycle_frame_cnt]

        larm_ee_pos = (
            self.sim.data.get_geom_xpos("larm_link2_mesh") - self.sim.data.qpos[0:3]
        )
        rarm_ee_pos = (
            self.sim.data.get_geom_xpos("rarm_link2_mesh") - self.sim.data.qpos[0:3]
        )
        lleg_ee_pos = (
            self.sim.data.get_geom_xpos("lleg_link4_mesh") - self.sim.data.qpos[0:3]
        )
        rleg_ee_pos = (
            self.sim.data.get_geom_xpos("rleg_link4_mesh") - self.sim.data.qpos[0:3]
        )
        current_ee_pos = np.concatenate(
            [rleg_ee_pos, lleg_ee_pos, rarm_ee_pos, larm_ee_pos], axis=0
        )  # NOTE local

        # mimic reward
        mimic_jnt_pos_reward = 0.65 * np.exp(
            # -2.0
            # -0.1
            -2.0  # TODO Optimize this
            * (
                np.linalg.norm(
                    ref_jnt_pos - self.sim.data.qpos.flat[7 : 7 + self.n_control_joints]
                )
                ** 2
            )
        )  # NOTE hyperparameter from original paper
        mimic_jnt_vel_reward = 0.1 * np.exp(
            # -0.1
            -0.1
            * (
                np.linalg.norm(
                    ref_jnt_vel - self.sim.data.qvel.flat[6 : 6 + self.n_control_joints]
                )
                ** 2
            )
        )  # NOTE hyperparameter from original paper
        mimic_ee_pos_reward = 0.15 * np.exp(
            -40.0 * (np.linalg.norm(ref_ee_pos - current_ee_pos) ** 2)
        )  # NOTE hyperparameter from original paper
        mimic_base_pos_reward = 0.1 * np.exp(
            -10.0 * (np.linalg.norm(self.sim.data.qpos.flat[0:3] - ref_base_pos) ** 2)
        )  # NOTE hyperparameter from original paper
        mimic_base_quat_reward = 0.1 * np.exp(
            -50.0 * (1 - np.dot(self.sim.data.qpos.flat[3:7], ref_base_quat))
        )  # NOTE not from original paper
        mimic_base_lin_vel_reward = 0.1 * np.exp(
            -1.0
            * (np.linalg.norm(self.sim.data.qvel.flat[0:3] - ref_base_lin_vel) ** 2)
        )  # NOTE not from original paper

        # print(
        #     "local time step : {}, loop count : {}, cycle time step : {}".format(
        #         self.local_frame_cnt, self.loop_cnt, self.cycle_frame_cnt
        #     )
        # )

        mimic_reward = (
            mimic_jnt_pos_reward
            + mimic_jnt_vel_reward
            + mimic_ee_pos_reward
            + mimic_base_pos_reward
            # + mimic_base_quat_reward
            + mimic_base_lin_vel_reward
        )
        task_reward = 0.0
        reward = mimic_reward + task_reward

        # if self.loop_cnt > 0 and (
        #     (self.sim.data.qpos[0] - self.ref_base_pos[self.init_motion_data_frame, 0])
        #     < 0.2
        # ):
        #     reward += 5.0

        # do simulation
        self.torque_max = 2.5
        self.kp = 10.0

        if self.control_type == "pd":
            target_jnt_pos = np.clip(action * np.pi, -np.pi, np.pi)
            current_jnt_pos = self.sim.data.qpos.flat[
                7 : 7 + self.n_control_joints
            ]  # NOTE no head joint
            current_jnt_vel = self.sim.data.qvel.flat[
                6 : 6 + self.n_control_joints
            ]  # NOTE no head joint
            torque = np.clip(
                self.kp * (target_jnt_pos - current_jnt_pos)
                - self.kd * current_jnt_vel,
                -self.torque_max,
                self.torque_max,
            )
        elif self.control_type == "torque":
            torque = np.clip(
                action * self.torque_max, -self.torque_max, self.torque_max
            )
        else:
            AssertionError("invalid control type")
        self.do_simulation(torque, self.frame_skip)

        self.episode_cnt += 1
        self.step_cnt += 1
        self.frame_cnt += 1

        # check done
        # check contact
        self.l_foot_contact = self.floor_contact_check(self.lleg_link4_geom_id)
        self.r_foot_contact = self.floor_contact_check(self.rleg_link4_geom_id)

        # self.terminal_contact = self.floor_contact_check(self.terminal_contact_geom_ids)
        self.terminal_contact = False
        for geom_id in self.terminal_contact_geom_ids:
            self.terminal_contact |= self.floor_contact_check(geom_id)
            break

        # if (self.frame_cnt > self.motion_cycle_frames) and (
        #     (self.sim.data.qpos[0] - self.ref_base_pos[self.init_motion_data_frame, 0])
        #     < 0.18
        # ):
        #     not_forward = True
        #     # TODO
        #     reward -= 10.0
        # else:
        #     not_forward = False

        if (self.frame_cnt > self.motion_cycle_frames * (self.loop_cnt + 1)) and (
            (self.sim.data.qpos[0] - self.ref_base_pos[self.init_motion_data_frame, 0])
            < (0.18 * (self.loop_cnt + 1))
        ):
            not_forward = True
            # TODO
            reward -= 10.0
        else:
            not_forward = False

        # done definition
        done = self.episode_cnt >= self.max_episode  # max episode
        if self.early_termination:
            done |= self.sim.data.qpos[2] < 0.2  # fallen
            done |= self.terminal_contact  # contact with ground other than feet
            done |= not_forward  # not forward

        # get observations
        self.set_buffer()
        obs = self._get_obs()

        # return step results
        self.episode_reward += reward

        self.episode_mimic_jnt_pos_reward += mimic_jnt_pos_reward
        self.episode_mimic_jnt_vel_reward += mimic_jnt_vel_reward
        self.episode_mimic_ee_pos_reward += mimic_ee_pos_reward
        self.episode_mimic_base_pos_reward += mimic_base_pos_reward
        self.episode_mimic_base_quat_reward += mimic_base_quat_reward
        self.episode_mimic_base_lin_vel_reward += mimic_base_lin_vel_reward

        if done:
            self.episode_cnt = 0
            self.buffer = None
            reward = 0.0
            print("-----------------------------")
            print("episode length : {}".format(self.frame_cnt))
            print("episode reward : {:.3f}".format(self.episode_reward))
            print(
                "mimic_jnt_pos_reward : {:.3f}".format(
                    self.episode_mimic_jnt_pos_reward / self.episode_reward
                )
            )
            print(
                "mimic_jnt_vel_reward : {:.3f}".format(
                    self.episode_mimic_jnt_vel_reward / self.episode_reward
                )
            )
            print(
                "mimic_ee_pos_reward : {:.3f}".format(
                    self.episode_mimic_ee_pos_reward / self.episode_reward
                )
            )
            print(
                "mimic_base_pos_reward : {:.3f}".format(
                    self.episode_mimic_base_pos_reward / self.episode_reward
                )
            )
            print(
                "mimic_base_quat_reward : {:.3f}".format(
                    self.episode_mimic_base_quat_reward / self.episode_reward
                )
            )
            print(
                "mimic_base_lin_vel_reward : {:.3f}".format(
                    self.episode_mimic_base_lin_vel_reward / self.episode_reward
                )
            )
            print("-----------------------------")
        return (
            obs,
            reward,
            done,
            dict(
                mimic_jnt_pos_reward=mimic_jnt_pos_reward,
                mimic_jnt_vel_reward=mimic_jnt_vel_reward,
                mimic_base_pos_reward=mimic_base_pos_reward,
                mimic_base_quat_reward=mimic_base_quat_reward,
                mimic_base_lin_vel_reward=mimic_base_lin_vel_reward,
                mimic_ee_pos_reward=mimic_ee_pos_reward,
                episode_reward=self.episode_reward,
            ),
        )

    def _get_obs(self):
        phase = np.array(
            [self.cycle_frame_cnt / self.motion_cycle_frames], dtype=np.float32
        )
        return np.concatenate(
            [np.array(obs, dtype=np.float32).flatten() for obs in self.buffer.values()]
            + [phase]
        )

    def _set_action_space(self):
        # bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        # low, high = bounds.T
        low = np.ones(16, dtype=np.float32) * -1.0
        high = np.ones(16, dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def reset_model(self):
        self.time = 0.0
        self.frame_cnt = 0
        self.episode_reward = 0.0

        self.episode_mimic_jnt_pos_reward = 0.0
        self.episode_mimic_jnt_vel_reward = 0.0
        self.episode_mimic_ee_pos_reward = 0.0
        self.episode_mimic_base_pos_reward = 0.0
        self.episode_mimic_base_quat_reward = 0.0
        self.episode_mimic_base_lin_vel_reward = 0.0

        self.init_motion_data_frame = np.random.randint(
            low=0, high=self.motion_cycle_frames
        )
        qpos = self.init_qpos  # base_pos [3], base_quat [4], jnt_pos [17]
        qvel = self.init_qvel
        qpos[0:3] = self.ref_base_pos[self.init_motion_data_frame, :]  # base_pos [3]
        qpos[3:7] = self.ref_base_quat[self.init_motion_data_frame, :]  # base_quat [4]
        qpos[7 : 7 + self.n_control_joints] = self.ref_jnt_pos[
            self.init_motion_data_frame, :
        ]  # jnt_pos without head jouint [16]
        qpos[-1] = 0.0  # for head joint [1]
        qvel[0:3] = self.ref_base_lin_vel[
            self.init_motion_data_frame, :
        ]  # base_lin_vel [3]
        qvel[3:6] = self.ref_base_ang_vel[
            self.init_motion_data_frame, :
        ]  # base_ang_vel [3]
        qvel[6 : 6 + self.n_control_joints] = self.ref_jnt_vel[
            self.init_motion_data_frame, :
        ]  # jnt_vel without head joint [16]
        qvel[-1] = 0.0  # for head joint [1]

        self.set_state(qpos, qvel)
        self.set_buffer()
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 1.0

    def render(self, *args, **kwargs):
        if self.viewer:
            size = [0.015] * 3

            ref_base_pos = self.ref_base_pos[
                self.cycle_frame_cnt
            ] + self.loop_cnt * np.array([self.ref_base_pos[-1, 0], 0.0, 0.0])

            self.viewer.add_marker(
                pos=ref_base_pos,  # Position
                label="base pos ref",  # Text beside the marker
                type=const.GEOM_SPHERE,  # Geomety type
                size=size,  # Size of the marker
                rgba=(1, 0, 1, 1),  # RGBA of the marker
            )  # RGBA of the marker

            self.viewer.add_marker(
                pos=self.ref_ee_pos[self.cycle_frame_cnt, 0:3]
                + ref_base_pos,  # Position
                label=" ",  # Text beside the marker
                type=const.GEOM_SPHERE,  # Geomety type
                size=size,  # Size of the marker
                rgba=(1, 0, 0, 1),  # Red
            )  # RGBA of the marker
            self.viewer.add_marker(
                pos=self.ref_ee_pos[self.cycle_frame_cnt, 3:6]
                + ref_base_pos,  # Position
                label=" ",  # Text beside the marker
                type=const.GEOM_SPHERE,  # Geomety type
                size=size,  # Size of the marker
                rgba=(0, 1, 0, 1),  # Green
            )  # RGBA of the marker
            self.viewer.add_marker(
                pos=self.ref_ee_pos[self.cycle_frame_cnt, 6:9]
                + ref_base_pos,  # Position
                label=" ",  # Text beside the marker
                type=const.GEOM_SPHERE,  # Geomety type
                size=size,  # Size of the marker
                rgba=(0, 0, 1, 1),  # Blue
            )  # RGBA of the marker
            self.viewer.add_marker(
                pos=self.ref_ee_pos[self.cycle_frame_cnt, 9:12]
                + ref_base_pos,  # Position
                label=" ",  # Text beside the marker
                type=const.GEOM_SPHERE,  # Geomety type
                size=size,  # Size of the marker
                rgba=(1, 1, 0, 1),  # Yellow
            )  # RGBA of the marker

        super(KHRMimicEnv, self).render(*args, **kwargs)


if __name__ == "__main__":
    env = KHRMimicEnv(test=True)
    env.reset()

    for i in range(10000):
        test_action = np.random.uniform(-1.0, 1.0, 16)
        obs, reward, done, info = env.step(test_action)
        if done:
            env.reset()
        env.render()

#!/usr/bin/env python3
import copy
import numpy as np
from gym import utils, spaces
from gym.envs.mujoco.mujoco_env import MujocoEnv
from mujoco_py.generated import const

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
    def __init__(self, test=False, max_step=20000000):
        self.is_params_set = False
        self.test = test
        self.max_step = max_step  # 20000000
        frame_skip = 10  # for 50 Hz control

        # TODO
        self.time = 0
        self.init_motion_data_frame = 0
        self.load_motion_data()

        model_path = "/home/oh/ros/lecture_ws/src/agent-system/lecture2023/student_projects/khr_mimic/models/KHR/KHR.xml"
        MujocoEnv.__init__(self, model_path, frame_skip=frame_skip)
        utils.EzPickle.__init__(self)

        print("sim dt", self.dt)

    def load_motion_data(self):
        motion_data = np.load(
            "/home/oh/ros/lecture_ws/src/agent-system/lecture2023/student_projects/khr_mimic/data/processed_07_08.npz",
            allow_pickle=True,
        )
        self.ref_base_pos = motion_data["ref_base_pos"]
        self.ref_base_quat = motion_data["ref_base_quat"]
        self.ref_jnt_pos = motion_data["ref_jnt_pos"]
        self.ref_base_lin_vel = motion_data["ref_base_lin_vel"]
        self.ref_base_ang_vel = motion_data["ref_base_ang_vel"]
        self.ref_jnt_vel = motion_data["ref_jnt_vel"]
        self.ref_ee_pos = motion_data["ref_ee_pos"].item()
        self.frame_duration = motion_data["frame_duration"]
        self.num_frames = motion_data["num_frames"]

        self.motion_cycle_frames = 150  # TODO
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
        self.accelerometer_id = self.model.sensor_name2id("accelerometer")
        self.gyro_id = self.model.sensor_name2id("gyro")
        self.framequat_id = self.model.sensor_name2id("framequat")
        self.velocimeter_id = self.model.sensor_name2id("velocimeter")
        self.framepos_id = self.model.sensor_name2id("framepos")

        self.torque_max = 2.5  # [Nm]
        self.kp = 10.0  # TODO
        self.kd = 0.03  # TODO

        self.buffer = None
        self.buffer_size = 6

        # variable for rl
        self.max_episode = 10000
        self.episode_cnt = 0  # local
        self.step_cnt = 0  # global

    def set_buffer(self):
        action = self.action
        base_pos_z = self.sim.data.qpos[2, np.newaxis]
        base_quat = self.sim.data.qpos[3:7]
        jnt_pos = self.sim.data.qpos[
            7 : 7 + self.n_control_joints
        ]  # NOTE without head joint
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

    def compute_interpolated_motion(self, time):
        # TODO
        return 0

    def step(self, action):  # action : joint torque (17)
        self.action = action[:-1]  # without head joint
        if not self.is_params_set:
            self.set_param()
            self.set_buffer()
            self.is_params_set = True

        # TODO
        current_frame = 0
        next_frame = current_frame + 1

        self.time += self.dt

        # check contact
        self.l_foot_contact = self.floor_contact_check(self.lleg_link4_geom_id)
        self.r_foot_contact = self.floor_contact_check(self.rleg_link4_geom_id)

        # self.terminal_contact = self.floor_contact_check(self.terminal_contact_geom_ids)
        self.terminal_contact = False
        for geom_id in self.terminal_contact_geom_ids:
            self.terminal_contact |= self.floor_contact_check(geom_id)
            break

        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)

        # do simulation
        # target_jnt_pos = np.clip(action * np.pi, -np.pi, np.pi)
        # target_jnt_pos[-1] = 0.0  # NOTE head joint is not used
        # current_jnt_pos = self.sim.data.qpos.flat[7:]
        # current_jnt_vel = self.sim.data.qvel.flat[6:]
        # torque = np.clip(
        #     self.kp * (target_jnt_pos - current_jnt_pos) - self.kd * current_jnt_vel,
        #     -self.torque_max,
        #     self.torque_max,
        # )
        torque = np.clip(action * self.torque_max, -self.torque_max, self.torque_max)
        self.do_simulation(torque, self.frame_skip)

        # reward definition
        ctrl_reward = 0.0  # restraint for joint action (torque)
        contact_reward = 0.0  # restraint for contact between ground and pose/support
        survive_reward = 0.0  # survive reward
        go_pos_reward = copy.deepcopy(self.sim.data.qpos[0])  # go forward reward

        # mimic reward
        mimic_qpos_reward = 0.0
        mimic_qvel_reward = 0.0
        mimic_base_reward = 0.0
        mimic_base_quat_reward = 0.0
        mimic_base_vel_reward = 0.0

        mimic_reward = (
            mimic_qpos_reward
            + mimic_qvel_reward
            + mimic_base_reward
            + mimic_base_quat_reward
            + mimic_base_vel_reward
        )

        reward = (
            survive_reward
            + mimic_reward
            + go_pos_reward
            # + ctrl_reward
            # + contact_reward
        )

        self.episode_cnt += 1
        self.step_cnt += 1

        # done definition
        done = self.episode_cnt >= self.max_episode  # max episode
        done |= self.sim.data.qpos[2] < 0.18  # fallen
        done |= self.terminal_contact  # contact with ground other than feet

        self.set_buffer()

        obs = self._get_obs()
        if done:
            self.episode_cnt = 0
            self.buffer = None
            reward = 0.0

        return (
            obs,
            reward,
            done,
            dict(
                ctrl_reward=ctrl_reward,
                contact_reward=contact_reward,
                survive_reward=survive_reward,
            ),
        )

    def _get_obs(self):
        phase = np.array(
            (
                self.init_motion_data_frame
                + self.time % self.motion_cycle_period / self.frame_duration
            )
            % self.motion_cycle_frames
            / self.motion_cycle_frames,
            dtype=np.float32,
        )
        phase = phase[np.newaxis]
        return np.concatenate(
            [np.array(obs, dtype=np.float32).flatten() for obs in self.buffer.values()]
            + [phase]
        )

    def _set_action_space(self):
        # bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        # low, high = bounds.T
        low = np.ones(17, dtype=np.float32) * -1.0
        high = np.ones(17, dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def reset_model(self):
        self.time = 0.0
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


if __name__ == "__main__":
    env = KHRMimicEnv(test=True)
    env.reset()

    for i in range(10000):
        test_action = np.random.uniform(-1.0, 1.0, 17)
        obs, reward, done, info = env.step(test_action)
        if done:
            env.reset()
        env.render()

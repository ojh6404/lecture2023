#!/usr/bin/env python3
import copy
import numpy as np
from gym import utils, spaces
from gym.envs.mujoco.mujoco_env import MujocoEnv

# import ramiel_utils


class KHRMimicEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, test=False, max_step=20000000):
        self.is_params_set = False
        self.test = test
        self.max_step = max_step # 20000000
        frame_skip = 10 # for 50 Hz control

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
        self.frame_duration = motion_data["frame_duration"]
        self.num_frames = motion_data["num_frames"]

        self.motion_cycle_frames = 150  # TODO
        self.motion_cycle_period = self.motion_cycle_frames * self.frame_duration

    def set_param(self):
        # get joint/pose id
        self.jnt_pos_indices = self.model.jnt_qposadr[1:]  # joint pos indices
        self.jnt_vel_indices = self.model.jnt_dofadr[1:]  # joint vel indices

        self.n_joints = len(self.jnt_pos_indices)

        # get geom id

        self.floor_geom_id = self.model.geom_name2id("floor")
        self.lleg_link4_geom_id = self.model.geom_name2id("lleg_link4_mesh")
        self.rleg_link4_geom_id = self.model.geom_name2id("rleg_link4_mesh")
        self.head_geom_id = self.model.geom_name2id("head_link0_mesh")

        self.terminal_contact_geom_ids = [self.model.geom_name2id(geom_names) for geom_names in self.model.geom_names if geom_names not in ["floor", "lleg_link4_mesh", "rleg_link4_mesh"]]

        # geom_ids = [self.model.geom_name2id(geom_names) for geom_names in self.model.geom_names]
        # for i in geom_ids:
        #     print(i, self.model.geom_id2name(i))

        # 0 floor
        # 1 body_link_mesh
        # 2 rleg_link0_mesh
        # 3 rleg_link1_mesh
        # 4 rleg_link2_mesh
        # 5 rleg_link3_mesh
        # 6 rleg_link4_mesh
        # 7 lleg_link0_mesh
        # 8 lleg_link1_mesh
        # 9 lleg_link2_mesh
        # 10 lleg_link3_mesh
        # 11 lleg_link4_mesh
        # 12 rarm_link0_mesh
        # 13 rarm_link1_mesh
        # 14 rarm_link2_mesh
        # 15 larm_link0_mesh
        # 16 larm_link1_mesh
        # 17 larm_link2_mesh
        # 18 head_link0_mesh

        # get sensor id
        # self.touch_sensor_id = self.model.sensor_name2id("contact_sensor")
        self.accelerometer_id = self.model.sensor_name2id("accelerometer")
        self.gyro_id = self.model.sensor_name2id("gyro")
        self.framequat_id = self.model.sensor_name2id("framequat")
        self.velocimeter_id = self.model.sensor_name2id("velocimeter")
        self.framepos_id = self.model.sensor_name2id("framepos")

        self.torque_max = 2.5  # [Nm]
        self.kp = 10.0
        self.kd = 0.03
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
        if not self.is_params_set:
            self.set_param()
            self.is_params_set = True

        # TODO
        current_frame = 0
        next_frame = current_frame + 1

        if self.max_step:
            step_rate = float(self.step_cnt) / self.max_step
        elif self.test:
            step_rate = self.default_step_rate

        self.time += self.dt


        # check contact
        self.l_foot_contact = self.floor_contact_check(self.lleg_link4_geom_id)
        self.r_foot_contact = self.floor_contact_check(self.rleg_link4_geom_id)

        # self.terminal_contact = self.floor_contact_check(self.terminal_contact_geom_ids)
        self.terminal_contact = False
        for geom_id in self.terminal_contact_geom_ids:
            self.terminal_contact |= self.floor_contact_check(geom_id)
            break

        # mujoco_py function that gets all geom ids


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
        """
        qpos
        0 : base_x, 1 : base_y, 2 : base_z
        3 : base_quat_w, 4 : base_quat_x, 5 : base_quat_y, 6 : base_quat_z
        7 : rleg_joint0, 8 : rleg_joint1, 9 : rleg_joint2, 10 : rleg_joint3, 11 : rleg_joint4
        12 : lleg_joint0, 13 : lleg_joint1, 14 : lleg_joint2, 15 : lleg_joint3, 16 : lleg_joint4
        17 : rarm_joint0, 18 : rarm_joint1, 19 : rarm_joint2
        20 : larm_joint0, 21 : larm_joint1, 22 : larm_joint2
        23 : head_joint0

        """
        torque = np.clip(action * self.torque_max, -self.torque_max, self.torque_max)
        # torque[-1] = 0.0 # head joint is not used
        self.do_simulation(torque, self.frame_skip)

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
        notdone = not self.terminal_contact
        notdone &= self.episode_cnt < self.max_episode
        if self.step_cnt == 1:
            done = False
        else:
            done = not notdone
        if self.sim.data.qpos[2] < 0.18:
            done |= True

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
            self.const_ext_qpos = self.const_qpos_rand * step_rate * np.random.randn(21)
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

        phase = np.array(
            (
                self.init_motion_data_frame
                + self.time % self.motion_cycle_period / self.frame_duration
            )
            % self.motion_cycle_frames
            / self.motion_cycle_frames
        )

        qpos = self.sim.data.qpos.flatten()  # base_pos [3], base_quat [4], jnt_pos [17]
        base_quat = qpos[3:7]
        jnt_pos = qpos[7:]
        base_pos_z = qpos[2, np.newaxis]
        qvel = self.sim.data.qvel.flatten()
        phase = phase[np.newaxis]

        # print("base_quat", base_quat.shape)
        # print("jnt_pos", jnt_pos.shape)
        # print("qvel", qvel.shape)
        # print("base_pos_z", base_pos_z)
        # print("phase", phase)
        # print("time", self.time)

        return np.concatenate([base_quat, jnt_pos, qvel, base_pos_z, phase])

        # return np.concatenate(
        #     [
        #         np.concatenate(self.prev_qpos),  # prev base quat + joint angles
        #         np.concatenate(self.prev_qvel),  # prev base quat vel + joint vels
        #         np.concatenate(self.prev_bvel),  # prev base linear vel
        #         np.concatenate(self.prev_action),  # prev action
        #     ]
        # )

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
        )  # 11

        slide_qpos_id = self.model.jnt_qposadr[self.model.joint_name2id("rleg_joint0")]
        roll_qpos_id = self.model.jnt_qposadr[self.model.joint_name2id("rleg_joint2")]
        pitch_qpos_id = self.model.jnt_qposadr[self.model.joint_name2id("rleg_joint1")]

        if self.max_step:
            step_rate = float(self.step_cnt) / self.max_step
        elif self.test:
            step_rate = self.default_step_rate

        qpos = self.init_qpos # base_pos [3], base_quat [4], jnt_pos [17]
        qpos[2] = 0.3  # TODO
        qpos[slide_qpos_id] = 0.874
        qpos[roll_qpos_id] = 0.03 * step_rate * np.random.randn(1)
        qpos[pitch_qpos_id] = 0.03 * step_rate * np.random.randn(1)
        qvel = self.init_qvel
        qpos[7:] = 0.0
        qvel[:] = 0.0
        self.set_state(qpos, qvel)

        # if (self.prev_qpos is None) and (self.prev_action is None):
        #     self.current_qpos = self.sim.data.qpos.flat[3:]
        #     self.current_qvel = self.sim.data.qvel.flat[3:]
        #     self.current_bvel = self.sim.data.qvel.flat[:3]
        #     self.prev_action = [np.zeros(3) for i in range(self.n_prev)]
        #     self.prev_qpos = [
        #         self.current_qpos + self.qpos_rand * step_rate * np.random.randn(21)
        #         for i in range(self.n_prev)
        #     ]
        #     self.prev_qvel = [
        #         self.current_qvel
        #         + self.qvel_rand
        #         * step_rate
        #         * np.abs(self.sim.data.qvel.flat[3:])
        #         * np.random.randn(6)
        #         for i in range(self.n_prev)
        #     ]
        #     self.prev_bvel = [
        #         self.current_bvel
        #         + self.bvel_rand
        #         * step_rate
        #         * np.abs(self.sim.data.qvel.flat[:3])
        #         * np.random.randn(3)
        #         for i in range(self.n_prev)
        #     ]

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 1.0


if __name__ == "__main__":
    env = KHRMimicEnv(test=True)
    env.reset()

    for i in range(10000):
        # env.step(np.zeros(17))
        test_action = np.random.uniform(-1.0, 1.0, 17)
        obs, reward, done, info = env.step(test_action)
        if done:
            env.reset()
        env.render()

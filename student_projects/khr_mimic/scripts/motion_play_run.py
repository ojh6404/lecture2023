#!/usr/bin/env python3

import numpy as np
import mujoco_py
from scipy.ndimage import gaussian_filter1d
from matplotlib import pyplot as plt
from mujoco_py.generated import const
from utils import *


def main():
    model_path = "/home/oh/ros/lecture_ws/src/agent-system/lecture2023/student_projects/khr_mimic/models/KHR/KHR.xml"
    model = mujoco_py.load_model_from_path(model_path)
    sim = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)
    step = 0

    motion_data = np.load(
        "/home/oh/ros/lecture_ws/src/agent-system/lecture2023/student_projects/khr_mimic/data/09_01.npz",
        allow_pickle=True,
    )

    retarget_frames = motion_data["retarget_frames"]
    ref_joint_pos = motion_data["ref_joint_pos"]
    robot_joint_indices = motion_data["robot_joint_indices"]
    non_fixed_joint_indices = motion_data["non_fixed_joint_indices"]
    frame_duration = motion_data["frame_duration"]

    num_frames = retarget_frames.shape[0]

    # LARM RARM LLEG RLEG in pybullet
    ref_base_pos = retarget_frames[:, 0:3]  # x, y, z [m]
    ref_base_quat = retarget_frames[:, 3:7]  # x, y, z, w
    ref_jnt_pos = retarget_frames[:, 7:]  # joint angles [rad]
    ref_larm_jnt_pos = ref_jnt_pos[:, 0:3]
    ref_rarm_jnt_pos = ref_jnt_pos[:, 3:6]
    ref_lleg_jnt_pos = ref_jnt_pos[:, 6:11]
    ref_rleg_jnt_pos = ref_jnt_pos[:, 11:16]

    # TODO
    ref_base_pos[:, 0] = 1.0 * ref_base_pos[:, 0]

    # RLEG LLEG RARM LARM in mujoco
    ref_jnt_pos = np.hstack(
        [ref_rleg_jnt_pos, ref_lleg_jnt_pos, ref_rarm_jnt_pos, ref_larm_jnt_pos]
    )

    Y_SYM = True
    Z_STABLE = True
    Z_OFFSET = 0.048
    FILTER = True
    CLIP = 10

    if FILTER:
        for i in range(ref_jnt_pos.shape[-1]):
            ref_jnt_pos[:, i] = gaussian_filter1d(
                ref_jnt_pos[:, i], 2, axis=-1, mode="nearest"
            )

    if Y_SYM:
        ref_base_pos[:, 1] = remove_bias(ref_base_pos[:, 1], remove_mean=True)
    if Z_STABLE:
        ref_base_pos[:, 2] = remove_bias(ref_base_pos[:, 2], remove_mean=False)
    if Z_OFFSET:
        ref_base_pos[:, 2] += Z_OFFSET

    if FILTER:
        for i in range(ref_base_quat.shape[-1]):
            ref_base_quat[:, i] = gaussian_filter1d(
                ref_base_quat[:, i], 2, axis=-1, mode="nearest"
            )
        ref_base_quat = ref_base_quat / np.expand_dims(
            np.linalg.norm(ref_base_quat, axis=-1), axis=-1
        )

    # print("robot_joint_indices: ", robot_joint_indices)
    # print("non_fixed_joint_indices: ", non_fixed_joint_indices)

    ref_base_lin_vel = compute_linear_velocity(
        ref_base_pos, frame_duration, level=1, gaussian_filter=FILTER
    )
    ref_base_ang_vel = compute_angular_velocity(
        ref_base_quat, frame_duration, gaussian_filter=FILTER
    )  # NOTE input ref_base_quat must be x, y, z, w
    ref_jnt_vel = compute_linear_velocity(
        ref_jnt_pos, frame_duration, level=1, gaussian_filter=FILTER
    )

    ref_base_quat = ref_base_quat[:, [3, 0, 1, 2]]  # convert to w, x, y, z

    # slower for motion debugging
    # sim.model.opt.timestep = 0.01
    sim.model.opt.timestep = frame_duration

    # motion data info
    print("num_frames: ", num_frames)
    print("frame_duration: ", frame_duration)
    print("ref_base_pos: ", ref_base_pos.shape)
    print("ref_base_quat: ", ref_base_quat.shape)
    print("ref_jnt_pos: ", ref_jnt_pos.shape)
    print("ref_base_lin_vel: ", ref_base_lin_vel.shape)
    print("ref_base_ang_vel: ", ref_base_ang_vel.shape)
    print("ref_jnt_vel: ", ref_jnt_vel.shape)

    ref_base_pos = ref_base_pos[CLIP:-CLIP, :]
    ref_base_quat = ref_base_quat[CLIP:-CLIP, :]
    ref_jnt_pos = ref_jnt_pos[CLIP:-CLIP, :]
    ref_base_lin_vel = ref_base_lin_vel[CLIP:-CLIP, :]
    ref_base_ang_vel = ref_base_ang_vel[CLIP:-CLIP, :]
    ref_jnt_vel = ref_jnt_vel[CLIP:-CLIP, :]
    num_frames -= CLIP * 2

    ref_ee_pos = np.zeros((num_frames, 12), dtype=np.float32)

    try:
        while True:
            state = sim.get_state()
            motion_qpos = np.zeros(
                24, dtype=np.float32
            )  # base_pos (3), base_quat (4), joint_pos (17)
            motion_qpos[0:3] = ref_base_pos[step, :]  # x, y, z [m]
            motion_qpos[3:7] = ref_base_quat[step, :]  # w, x, y, z
            motion_qpos[7:23] = ref_jnt_pos[step, :]  # joint angles [rad]
            state = mujoco_py.MjSimState(
                state.time,
                motion_qpos,
                state.qvel,
                state.act,
                state.udd_state,
            )

            # code for get end effector position of lleg_link4...
            # sim.set_state(state)
            # sim.forward()
            # sim.step()
            # print(sim.data.get_body_xpos("lleg_link4"))

            # get end effector geom position
            ref_ee_pos[step, 0:3] = (
                sim.data.get_geom_xpos("rleg_link4_mesh") - sim.data.qpos[0:3]
            )
            ref_ee_pos[step, 3:6] = (
                sim.data.get_geom_xpos("lleg_link4_mesh") - sim.data.qpos[0:3]
            )
            ref_ee_pos[step, 6:9] = (
                sim.data.get_geom_xpos("rarm_link2_mesh") - sim.data.qpos[0:3]
            )
            ref_ee_pos[step, 9:12] = (
                sim.data.get_geom_xpos("larm_link2_mesh") - sim.data.qpos[0:3]
            )

            size = [0.015] * 3

            sim.set_state(state)
            sim.forward()
            sim.step()
            viewer.add_marker(
                pos=ref_ee_pos[step, 0:3] + sim.data.qpos[0:3],  # Position
                label=" ",  # Text beside the marker
                type=const.GEOM_SPHERE,  # Geomety type
                size=size,  # Size of the marker
                rgba=(1, 0, 0, 1),
            )  # RGBA of the marker
            viewer.add_marker(
                pos=ref_ee_pos[step, 3:6] + sim.data.qpos[0:3],  # Position
                label=" ",  # Text beside the marker
                type=const.GEOM_SPHERE,  # Geomety type
                size=size,  # Size of the marker
                rgba=(0, 1, 0, 1),
            )  # RGBA of the marker
            viewer.add_marker(
                pos=ref_ee_pos[step, 6:9] + sim.data.qpos[0:3],  # Position
                label=" ",  # Text beside the marker
                type=const.GEOM_SPHERE,  # Geomety type
                size=size,  # Size of the marker
                rgba=(0, 0, 1, 1),
            )  # RGBA of the marker
            viewer.add_marker(
                pos=ref_ee_pos[step, 9:12] + sim.data.qpos[0:3],  # Position
                label=" ",  # Text beside the marker
                type=const.GEOM_SPHERE,  # Geomety type
                size=size,  # Size of the marker
                rgba=(1, 1, 0, 1),
            )  # RGBA of the marker
            viewer.render()
            step += 1
            if step == num_frames:
                step = 0
    except KeyboardInterrupt:
        np.savez(
            "processed_09_01.npz",
            ref_base_pos=ref_base_pos,
            ref_base_quat=ref_base_quat,
            ref_jnt_pos=ref_jnt_pos,
            ref_jnt_vel=ref_jnt_vel,
            ref_base_lin_vel=ref_base_lin_vel,
            ref_base_ang_vel=ref_base_ang_vel,
            ref_ee_pos=ref_ee_pos,
            frame_duration=frame_duration,
            num_frames=num_frames,
        )

        total_time = num_frames * frame_duration
        dt = 0.02
        blended_num_frames = int(total_time / dt)
        blended_ref_base_pos = np.zeros((blended_num_frames, 3), dtype=np.float32)
        blended_ref_base_quat = np.zeros((blended_num_frames, 4), dtype=np.float32)
        blended_ref_jnt_pos = np.zeros((blended_num_frames, 16), dtype=np.float32)
        blended_ref_jnt_vel = np.zeros((blended_num_frames, 16), dtype=np.float32)
        blended_ref_base_lin_vel = np.zeros((blended_num_frames, 3), dtype=np.float32)
        blended_ref_base_ang_vel = np.zeros((blended_num_frames, 3), dtype=np.float32)
        blended_ref_ee_pos = np.zeros((blended_num_frames, 12), dtype=np.float32)

        for i in range(blended_num_frames):
            time = i * dt
            frame_idx0, frame_idx1, blend = calc_frame_blend(
                time, total_time, num_frames, frame_duration
            )
            blended_ref_base_pos[i, :] = (1 - blend) * ref_base_pos[
                frame_idx0, :
            ] + blend * ref_base_pos[frame_idx1, :]

            # blended_ref_base_quat[i, :] = slerp(
            #     ref_base_quat[frame_idx0, :], ref_base_quat[frame_idx1, :], blend
            # )

            blended_ref_base_quat[i, :] = quat_normalize(
                (1 - blend) * ref_base_quat[frame_idx0, [1, 2, 3, 0]]
                + blend * ref_base_quat[frame_idx1, [1, 2, 3, 0]]
            )[[3, 0, 1, 2]]

            blended_ref_jnt_pos[i, :] = (1 - blend) * ref_jnt_pos[
                frame_idx0, :
            ] + blend * ref_jnt_pos[frame_idx1, :]
            blended_ref_jnt_vel[i, :] = (1 - blend) * ref_jnt_vel[
                frame_idx0, :
            ] + blend * ref_jnt_vel[frame_idx1, :]
            blended_ref_base_lin_vel[i, :] = (1 - blend) * ref_base_lin_vel[
                frame_idx0, :
            ] + blend * ref_base_lin_vel[frame_idx1, :]
            blended_ref_base_ang_vel[i, :] = (1 - blend) * ref_base_ang_vel[
                frame_idx0, :
            ] + blend * ref_base_ang_vel[frame_idx1, :]
            blended_ref_ee_pos[i, :] = (1 - blend) * ref_ee_pos[
                frame_idx0, :
            ] + blend * ref_ee_pos[frame_idx1, :]

        # # NOTE
        # blended_ref_jnt_pos[:, 0] = np.clip(
        #     blended_ref_jnt_pos[:, 0], a_min=None, a_max=0.0
        # )
        # blended_ref_jnt_pos[:, 5] = np.clip(
        #     blended_ref_jnt_pos[:, 5], a_min=0, a_max=None
        # )

        np.savez(
            "blended_processed_09_01.npz",
            ref_base_pos=blended_ref_base_pos,
            ref_base_quat=blended_ref_base_quat,
            ref_jnt_pos=blended_ref_jnt_pos,
            ref_jnt_vel=blended_ref_jnt_vel,
            ref_base_lin_vel=blended_ref_base_lin_vel,
            ref_base_ang_vel=blended_ref_base_ang_vel,
            ref_ee_pos=blended_ref_ee_pos,
            frame_duration=dt,
            num_frames=blended_num_frames,
        )

        time_seq = np.arange(0, num_frames) * frame_duration
        blended_time_seq = np.arange(0, blended_num_frames) * dt

        print("time_seq", time_seq.shape)
        print("blended_time_seq", blended_time_seq.shape)
        print("ref_base_pos", ref_base_pos.shape)
        print("blended_ref_base_pos", blended_ref_base_pos.shape)

        plt.figure()
        plt.title("ref_base_pos")
        # plt.plot(time_seq, ref_base_lin_vel[:, 0], label="x")
        # plt.plot(time_seq, ref_base_lin_vel[:, 1], label="y")
        # plt.plot(time_seq, ref_base_lin_vel[:, 2], label="z")
        # plt.plot(time_seq, retarget_frames[CLIP:-CLIP, 0], label="x")
        # plt.plot(time_seq, retarget_frames[CLIP:-CLIP, 1], label="y")
        # plt.plot(time_seq, retarget_frames[CLIP:-CLIP, 2], label="z")
        plt.plot(blended_time_seq, blended_ref_base_pos[:, 0], label="x")
        plt.plot(blended_time_seq, blended_ref_base_pos[:, 1], label="y")
        plt.plot(blended_time_seq, blended_ref_base_pos[:, 2], label="z")
        plt.legend()

        plt.figure()
        plt.title("ref_base_quat")
        # plt.plot(time_seq, ref_base_quat[:, 0], label="w")
        # plt.plot(time_seq, ref_base_quat[:, 1], label="x")
        # plt.plot(time_seq, ref_base_quat[:, 2], label="y")
        # plt.plot(time_seq, ref_base_quat[:, 3], label="z")
        # plt.plot(time_seq, retarget_frames[CLIP:-CLIP, 6], label="w")
        # plt.plot(time_seq, retarget_frames[CLIP:-CLIP, 3], label="x")
        # plt.plot(time_seq, retarget_frames[CLIP:-CLIP, 4], label="y")
        # plt.plot(time_seq, retarget_frames[CLIP:-CLIP, 5], label="z")
        plt.plot(blended_time_seq, blended_ref_base_quat[:, 0], label="w")
        plt.plot(blended_time_seq, blended_ref_base_quat[:, 1], label="x")
        plt.plot(blended_time_seq, blended_ref_base_quat[:, 2], label="y")
        plt.plot(blended_time_seq, blended_ref_base_quat[:, 3], label="z")
        plt.legend()

        plt.figure("ref_base_lin_vel")
        plt.title("ref_base_lin_vel")
        # plt.plot(time_seq, ref_base_lin_vel[:, 0], label="x")
        # plt.plot(time_seq, ref_base_lin_vel[:, 1], label="y")
        # plt.plot(time_seq, ref_base_lin_vel[:, 2], label="z")
        plt.plot(blended_time_seq, blended_ref_base_lin_vel[:, 0], label="x")
        plt.plot(blended_time_seq, blended_ref_base_lin_vel[:, 1], label="y")
        plt.plot(blended_time_seq, blended_ref_base_lin_vel[:, 2], label="z")
        plt.legend()

        plt.figure()
        plt.title("ref_base_ang_vel")
        # plt.plot(time_seq, ref_base_ang_vel[:, 0], label="x")
        # plt.plot(time_seq, ref_base_ang_vel[:, 1], label="y")
        # plt.plot(time_seq, ref_base_ang_vel[:, 2], label="z")
        plt.plot(blended_time_seq, blended_ref_base_ang_vel[:, 0], label="x")
        plt.plot(blended_time_seq, blended_ref_base_ang_vel[:, 1], label="y")
        plt.plot(blended_time_seq, blended_ref_base_ang_vel[:, 2], label="z")
        plt.legend()

        plt.figure()
        plt.title("ref_jnt_pos")
        # plt.plot(time_seq, ref_jnt_pos[:, 0], label="rleg-crotch-r")
        # plt.plot(time_seq, ref_jnt_pos[:, 1], label="rleg-crotch-p")
        # plt.plot(time_seq, ref_jnt_pos[:, 2], label="rleg-knee-p")
        # plt.plot(time_seq, ref_jnt_pos[:, 3], label="rleg-ankle-p")
        # plt.plot(time_seq, ref_jnt_pos[:, 4], label="rleg-ankle-r")
        plt.plot(blended_time_seq, blended_ref_jnt_pos[:, 0], label="blend_rleg_0")
        plt.plot(blended_time_seq, blended_ref_jnt_pos[:, 1], label="blend_rleg_1")
        plt.plot(blended_time_seq, blended_ref_jnt_pos[:, 2], label="blend_rleg_2")
        plt.plot(blended_time_seq, blended_ref_jnt_pos[:, 3], label="blend_rleg_3")
        plt.plot(blended_time_seq, blended_ref_jnt_pos[:, 4], label="blend_rleg_4")

        plt.figure()
        plt.plot(ref_jnt_pos[:, 0], label="rleg-crotch-r")
        # plt.plot(blended_ref_jnt_pos[:, 0], label="blend_rleg_0")
        # plt.plot(blended_ref_jnt_pos[:, 1], label="blend_rleg_1")
        # plt.plot(blended_ref_jnt_pos[:, 2], label="blend_rleg_2")
        # plt.plot(blended_ref_jnt_pos[:, 3], label="blend_rleg_3")
        # plt.plot(blended_ref_jnt_pos[:, 4], label="blend_rleg_4")

        # plt.plot(ref_jnt_vel[:, 1], label="rleg_1")
        # plt.plot(ref_jnt_vel[:, 2], label="rleg_2")
        # plt.plot(ref_jnt_vel[:, 3], label="rleg_3")
        # plt.plot(ref_jnt_vel[:, 4], label="rleg_4")
        # plt.plot(ref_jnt_vel[:, 5], label="lleg_0")
        # plt.plot(ref_jnt_vel[:, 6], label="lleg_1")
        # plt.plot(ref_jnt_vel[:, 7], label="lleg_2")
        # plt.plot(ref_jnt_vel[:, 8], label="lleg_3")
        # plt.plot(ref_jnt_vel[:, 9], label="lleg_4")
        # plt.plot(ref_jnt_vel[:, 10], label="rarm_0")
        # plt.plot(ref_jnt_vel[:, 11], label="rarm_1")
        # plt.plot(ref_jnt_vel[:, 12], label="rarm_2")
        # plt.plot(ref_jnt_vel[:, 13], label="larm_0")
        # plt.plot(ref_jnt_vel[:, 14], label="larm_1")
        # plt.plot(ref_jnt_vel[:, 15], label="larm_2")
        plt.legend()

        plt.show()

        print("processed motion saved")

        step = 0
        while True:
            state = sim.get_state()
            motion_qpos = np.zeros(
                24, dtype=np.float32
            )  # base_pos (3), base_quat (4), joint_pos (17)
            motion_qpos[0:3] = blended_ref_base_pos[step, :]  # x, y, z [m]
            motion_qpos[3:7] = blended_ref_base_quat[step, :]  # w, x, y, z
            motion_qpos[7:23] = blended_ref_jnt_pos[step, :]  # joint angles [rad]
            state = mujoco_py.MjSimState(
                state.time,
                motion_qpos,
                state.qvel,
                state.act,
                state.udd_state,
            )

            # code for get end effector position of lleg_link4...
            # sim.set_state(state)
            # sim.forward()
            # sim.step()
            # print(sim.data.get_body_xpos("lleg_link4"))

            # get end effector geom position
            blended_ref_ee_pos[step, 0:3] = (
                sim.data.get_geom_xpos("rleg_link4_mesh") - sim.data.qpos[0:3]
            )
            blended_ref_ee_pos[step, 3:6] = (
                sim.data.get_geom_xpos("lleg_link4_mesh") - sim.data.qpos[0:3]
            )
            blended_ref_ee_pos[step, 6:9] = (
                sim.data.get_geom_xpos("rarm_link2_mesh") - sim.data.qpos[0:3]
            )
            blended_ref_ee_pos[step, 9:12] = (
                sim.data.get_geom_xpos("larm_link2_mesh") - sim.data.qpos[0:3]
            )

            size = [0.015] * 3

            sim.set_state(state)
            sim.forward()
            sim.step()
            # viewer.add_marker(
            #     pos=blended_ref_ee_pos[step, 0:3] + sim.data.qpos[0:3],  # Position
            #     label=" ",  # Text beside the marker
            #     type=const.GEOM_SPHERE,  # Geomety type
            #     size=size,  # Size of the marker
            #     rgba=(1, 0, 0, 1),
            # )  # RGBA of the marker
            # viewer.add_marker(
            #     pos=blended_ref_ee_pos[step, 3:6] + sim.data.qpos[0:3],  # Position
            #     label=" ",  # Text beside the marker
            #     type=const.GEOM_SPHERE,  # Geomety type
            #     size=size,  # Size of the marker
            #     rgba=(0, 1, 0, 1),
            # )  # RGBA of the marker
            # viewer.add_marker(
            #     pos=blended_ref_ee_pos[step, 6:9] + sim.data.qpos[0:3],  # Position
            #     label=" ",  # Text beside the marker
            #     type=const.GEOM_SPHERE,  # Geomety type
            #     size=size,  # Size of the marker
            #     rgba=(0, 0, 1, 1),
            # )  # RGBA of the marker
            # viewer.add_marker(
            #     pos=blended_ref_ee_pos[step, 9:12] + sim.data.qpos[0:3],  # Position
            #     label=" ",  # Text beside the marker
            #     type=const.GEOM_SPHERE,  # Geomety type
            #     size=size,  # Size of the marker
            #     rgba=(1, 1, 0, 1),
            # )  # RGBA of the marker
            viewer.render()
            step += 1
            if step == blended_num_frames:
                step = 0


if __name__ == "__main__":
    main()

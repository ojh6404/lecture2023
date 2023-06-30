#!/usr/bin/env python3
import numpy as np
import mujoco_py
from scipy.ndimage import gaussian_filter1d
from scipy import stats


def remove_bias(biased_data, remove_mean=True):
    x = np.arange(biased_data.shape[0])
    linear_regression = stats.linregress(x, biased_data)
    if remove_mean:
        mean = 0.0
    else:
        mean = np.mean(biased_data)
    result = (
        biased_data - (linear_regression.slope * x + linear_regression.intercept) + mean
    )
    return result


def main():
    model_path = "/home/oh/ros/lecture_ws/src/agent-system/lecture2023/student_projects/khr_mimic/models/KHR/KHR.xml"
    model = mujoco_py.load_model_from_path(model_path)
    sim = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)
    step = 0

    motion_data = np.load(
        "/home/oh/ros/lecture_ws/src/agent-system/lecture2023/student_projects/khr_mimic/data/07_08.npz",
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
    ref_base_quat = ref_base_quat[:, [3, 0, 1, 2]]  # w, x, y, z
    ref_jnt_pos = retarget_frames[:, 7:]  # joint angles [rad]
    ref_larm_jnt_pos = ref_jnt_pos[:, 0:3]
    ref_rarm_jnt_pos = ref_jnt_pos[:, 3:6]
    ref_lleg_jnt_pos = ref_jnt_pos[:, 6:11]
    ref_rleg_jnt_pos = ref_jnt_pos[:, 11:16]

    Y_SYM = True
    Z_STABLE = True
    Z_OFFSET = 0.065
    FILTER = True

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

    # motion data info
    print("num_frames: ", num_frames)
    print("frame_duration: ", frame_duration)
    print("ref_base_pos: ", ref_base_pos.shape)
    print("ref_base_quat: ", ref_base_quat.shape)
    print("ref_joint_pos: ", ref_joint_pos.shape)
    # print("robot_joint_indices: ", robot_joint_indices)
    # print("non_fixed_joint_indices: ", non_fixed_joint_indices)

    # RLEG LLEG RARM LARM in mujoco
    ref_jnt_pos = np.hstack(
        [ref_rleg_jnt_pos, ref_lleg_jnt_pos, ref_rarm_jnt_pos, ref_larm_jnt_pos]
    )

    # slower for motion debugging
    # sim.model.opt.timestep = 0.01
    sim.model.opt.timestep = frame_duration

    np.savez(
        "processed_07_08.npz",
        ref_base_pos=ref_base_pos,
        ref_base_quat=ref_base_quat,
        ref_jnt_pos=ref_jnt_pos,
        frame_duration=frame_duration,
        num_frames=num_frames,
    )

    while True:
        old_state = sim.get_state()
        motion_qpos = np.zeros(
            24, dtype=np.float32
        )  # base_pos (3), base_quat (4), joint_pos (17)
        motion_qpos[0:3] = ref_base_pos[step, :]  # x, y, z [m]
        motion_qpos[3:7] = ref_base_quat[step, :]  # w, x, y, z
        motion_qpos[7:23] = ref_jnt_pos[step, :]  # joint angles [rad]
        new_state = mujoco_py.MjSimState(
            old_state.time,
            motion_qpos,
            old_state.qvel,
            old_state.act,
            old_state.udd_state,
        )
        sim.set_state(new_state)
        sim.forward()
        sim.step()
        viewer.render()
        step += 1
        if step == num_frames:
            step = 0


if __name__ == "__main__":
    main()

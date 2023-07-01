#!/usr/bin/env python3
import numpy as np
import mujoco_py
from scipy.ndimage import gaussian_filter1d
from scipy import stats
from matplotlib import pyplot as plt
from mujoco_py.generated import const


def quat_pos(x):
    """
    make all the real part of the quaternion positive
    """
    q = x
    z = np.float32(q[..., 3:] < 0)  # 1 if negative, 0 if positive
    q = (1 - 2 * z) * q  # if negative, multiply by -1
    return q


def quat_abs(x):
    """
    quaternion norm (unit quaternion represents a 3D rotation, which has norm of 1)
    """
    x = np.linalg.norm(x, axis=-1)
    return x


def quat_unit(x):
    """
    normalized quaternion with norm of 1
    """
    norm = np.expand_dims(quat_abs(x), axis=-1)
    return x / np.clip(norm, a_min=1e-9, a_max=None)


def quat_normalize(q):
    """
    Construct 3D rotation from quaternion (the quaternion needs not to be normalized).
    """
    q = quat_unit(quat_pos(q))  # normalized to positive and unit quaternion
    return q


def quat_mul(a, b):
    """
    quaternion multiplication, x,y,z,w order
    """
    x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return np.stack([x, y, z, w], axis=-1)


def quat_mul_norm(x, y):
    """
    Combine two set of 3D rotations together using \**\* operator. The shape needs to be
    broadcastable
    """
    return quat_normalize(quat_mul(x, y))


def quat_identity(shape):
    """
    Construct 3D identity rotation given shape
    """
    shape = list(shape)
    w = np.ones(shape + [1])
    xyz = np.zeros(shape + [3])
    q = np.concatenate([xyz, w], axis=-1)
    return q


def quat_conjugate(x):
    """
    quaternion with its imaginary part negated
    """
    return np.concatenate([-x[..., :3], x[..., 3:]], axis=-1)


def quat_inverse(x):
    """
    The inverse of the rotation
    """
    return quat_conjugate(x)


def quat_angle_axis(x):
    """
    The (angle, axis) representation of the rotation. The axis is normalized to unit length.
    The angle is guaranteed to be between [0, pi].
    """
    s = 2 * (x[..., 3] ** 2) - 1
    angle = np.arccos(np.clip(s, -1, 1))  # just to be safe
    axis = x[..., :3]
    axis /= np.clip(
        np.linalg.norm(axis, axis=-1, keepdims=True), a_min=1e-9, a_max=None
    )
    return angle, axis


def compute_linear_velocity(f, dt, level=6, gaussian_filter=True):
    dfdt = np.zeros_like(f)
    df = f[1:, :] - f[:-1, :]
    dfdt[:-1, :] = df / dt
    dfdt[-1, :] = dfdt[-2, :]
    if gaussian_filter:
        dfdt = gaussian_filter1d(dfdt, level, axis=-2, mode="nearest")
    return dfdt


def compute_angular_velocity(r, time_delta, gaussian_filter=True):
    diff_quat_data = quat_identity(r.shape[:-1])
    diff_quat_data[:-1] = quat_mul_norm(r[1:, :], quat_inverse(r[:-1, :]))

    diff_quat_data[-1] = diff_quat_data[-2]
    diff_angle, diff_axis = quat_angle_axis(diff_quat_data)
    angular_velocity = diff_axis * np.expand_dims(diff_angle, axis=-1) / time_delta
    if gaussian_filter:
        angular_velocity = gaussian_filter1d(
            angular_velocity, 2, axis=-2, mode="nearest"
        )
    return angular_velocity

    # diff_quat_data[:-1] =


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
    ref_jnt_pos = retarget_frames[:, 7:]  # joint angles [rad]
    ref_larm_jnt_pos = ref_jnt_pos[:, 0:3]
    ref_rarm_jnt_pos = ref_jnt_pos[:, 3:6]
    ref_lleg_jnt_pos = ref_jnt_pos[:, 6:11]
    ref_rleg_jnt_pos = ref_jnt_pos[:, 11:16]

    # RLEG LLEG RARM LARM in mujoco
    ref_jnt_pos = np.hstack(
        [ref_rleg_jnt_pos, ref_lleg_jnt_pos, ref_rarm_jnt_pos, ref_larm_jnt_pos]
    )

    Y_SYM = True
    Z_STABLE = True
    Z_OFFSET = 0.065
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

    # plt.figure()
    # plt.title("ref_base_pos")
    # plt.plot(ref_base_lin_vel[:, 0], label="x")
    # plt.plot(ref_base_lin_vel[:, 1], label="y")
    # plt.plot(ref_base_lin_vel[:, 2], label="z")
    # plt.legend()

    # plt.figure()
    # plt.title("ref_base_ang_vel")
    # plt.plot(ref_base_ang_vel[:, 0], label="x")
    # plt.plot(ref_base_ang_vel[:, 1], label="y")
    # plt.plot(ref_base_ang_vel[:, 2], label="z")
    # plt.legend()

    # plt.figure()
    # plt.title("ref_jnt_pos")
    # plt.plot(ref_jnt_vel[:, 0], label="rleg_0")
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
    # plt.legend()

    # plt.show()

    ref_ee_pos = dict()
    ref_ee_pos["larm"] = np.zeros((num_frames, 3), dtype=np.float32)
    ref_ee_pos["rarm"] = np.zeros((num_frames, 3), dtype=np.float32)
    ref_ee_pos["lleg"] = np.zeros((num_frames, 3), dtype=np.float32)
    ref_ee_pos["rleg"] = np.zeros((num_frames, 3), dtype=np.float32)

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
            ref_ee_pos["lleg"][step, :] = sim.data.get_geom_xpos("lleg_link4_mesh")
            ref_ee_pos["rleg"][step, :] = sim.data.get_geom_xpos("rleg_link4_mesh")
            ref_ee_pos["larm"][step, :] = sim.data.get_geom_xpos("larm_link2_mesh")
            ref_ee_pos["rarm"][step, :] = sim.data.get_geom_xpos("rarm_link2_mesh")

            size = [0.015] * 3

            sim.set_state(state)
            sim.forward()
            sim.step()
            viewer.add_marker(
                pos=ref_ee_pos["lleg"][step, :],  # Position
                label=" ",  # Text beside the marker
                type=const.GEOM_SPHERE,  # Geomety type
                size=size,  # Size of the marker
                rgba=(1, 0, 0, 1),
            )  # RGBA of the marker
            viewer.add_marker(
                pos=ref_ee_pos["rleg"][step, :],  # Position
                label=" ",  # Text beside the marker
                type=const.GEOM_SPHERE,  # Geomety type
                size=size,  # Size of the marker
                rgba=(0, 1, 0, 1),
            )  # RGBA of the marker
            viewer.add_marker(
                pos=ref_ee_pos["larm"][step, :],  # Position
                label=" ",  # Text beside the marker
                type=const.GEOM_SPHERE,  # Geomety type
                size=size,  # Size of the marker
                rgba=(0, 0, 1, 1),
            )  # RGBA of the marker
            viewer.add_marker(
                pos=ref_ee_pos["rarm"][step, :],  # Position
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
            "processed_07_08.npz",
            ref_base_pos=ref_base_pos,
            ref_base_quat=ref_base_quat,
            ref_jnt_pos=ref_jnt_pos,
            ref_base_lin_vel=ref_base_lin_vel,
            ref_base_ang_vel=ref_base_ang_vel,
            ref_jnt_vel=ref_jnt_vel,
            ref_ee_pos=ref_ee_pos,
            frame_duration=frame_duration,
            num_frames=num_frames,
        )
        print("processed motion saved")


if __name__ == "__main__":
    main()

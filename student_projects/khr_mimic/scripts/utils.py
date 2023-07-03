#!/usr/bin/env python3

import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d


def calc_frame_blend(time, len, num_frames, dt):
    phase = np.clip(time / len, 0.0, 1.0)
    frame_idx0 = (phase * (num_frames - 1)).astype(np.int32)
    frame_idx1 = np.minimum(frame_idx0 + 1, num_frames - 1)
    blend = (time - frame_idx0 * dt) / dt
    return frame_idx0, frame_idx1, blend


def slerp(q0, q1, blend):
    # qx, qy, qz, qw = 0, 1, 2, 3 # for xyzw
    qx, qy, qz, qw = 1, 2, 3, 0  # for wxyz

    cos_half_theta = (
        q0[..., qw] * q1[..., qw]
        + q0[..., qx] * q1[..., qx]
        + q0[..., qy] * q1[..., qy]
        + q0[..., qz] * q1[..., qz]
    )

    neg_mask = cos_half_theta < 0
    q1 = q1.copy()
    q1[neg_mask] = -q1[neg_mask]
    cos_half_theta = np.abs(cos_half_theta)
    cos_half_theta = np.expand_dims(cos_half_theta, axis=-1)

    half_theta = np.arccos(cos_half_theta)
    sin_half_theta = np.sqrt(1.0 - cos_half_theta * cos_half_theta)

    ratioA = np.sin((1 - blend) * half_theta) / sin_half_theta
    ratioB = np.sin(blend * half_theta) / sin_half_theta

    new_q_x = ratioA * q0[..., qx : qx + 1] + ratioB * q1[..., qx : qx + 1]
    new_q_y = ratioA * q0[..., qy : qy + 1] + ratioB * q1[..., qy : qy + 1]
    new_q_z = ratioA * q0[..., qz : qz + 1] + ratioB * q1[..., qz : qz + 1]
    new_q_w = ratioA * q0[..., qw : qw + 1] + ratioB * q1[..., qw : qw + 1]

    cat_dim = len(new_q_w.shape) - 1
    new_q = np.concatenate([new_q_x, new_q_y, new_q_z, new_q_w], axis=cat_dim)

    new_q = np.where(np.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
    new_q = np.where(np.abs(cos_half_theta) >= 1, q0, new_q)

    return new_q


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

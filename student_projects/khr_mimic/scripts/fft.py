#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main():
    motion_data = np.load(
        "/home/oh/ros/lecture_ws/src/agent-system/lecture2023/student_projects/khr_mimic/data/processed_07_08.npz",
        allow_pickle=True,
    )
    ref_base_pos = motion_data["ref_base_pos"]
    ref_base_quat = motion_data["ref_base_quat"]
    ref_jnt_pos = motion_data["ref_jnt_pos"]
    frame_duration = motion_data["frame_duration"]
    num_frames = motion_data["num_frames"]

    time = np.arange(0, frame_duration * num_frames, frame_duration)
    assert len(time) == num_frames

    motion_jnt_data = ref_jnt_pos[:, 1]

    plt.plot(motion_jnt_data)
    plt.show()

    motion_fft = np.fft.fft(motion_jnt_data)
    amplitude = abs(motion_fft) * 2 / num_frames
    frequency = np.fft.fftfreq(len(motion_jnt_data), frame_duration)

    fft_freq = frequency.copy()
    peak_index = amplitude[: int(len(amplitude) / 2)].argsort()[-3:]
    print("peak_index: ", peak_index)
    peak_freq = fft_freq[peak_index][-1]
    main_cycle_period = 1 / peak_freq
    main_cycle_step = int(main_cycle_period / frame_duration)
    print("peak_freq: ", peak_freq)
    print("main cycle: ", main_cycle_period)
    print("main cycle step: ", main_cycle_step)

    plt.xlim(0, 50)
    plt.stem(frequency, amplitude)
    plt.grid(True)
    plt.show()

    top3_freq = fft_freq[peak_index]
    top3_amp = amplitude[peak_index]
    print("top3_freq: ", top3_freq)

    fft_3x = np.zeros_like(motion_fft)

    print(fft_3x.shape)
    fft_3x[peak_index[0]] = motion_fft[peak_index[0]]
    fft_3x[peak_index[1]] = motion_fft[peak_index[1]]
    fft_3x[peak_index[2]] = motion_fft[peak_index[2]]

    filtered_motion = 2 * np.fft.ifft(fft_3x)
    cycle = round(1 / frame_duration / peak_freq)

    plt.plot(filtered_motion)
    plt.plot(motion_jnt_data)
    plt.show()

    # cycle_step = 186 - 39 = 147, 338 - 186 = 152 ~ 150
    # cycle_period = 147 * 0.00833 = 1.225

    # fft_1x = motion_fft.copy()
    # print("test")
    # print(fft_freq)
    # print(fft_freq == top3_freq[0])
    # print(fft_freq == top3_freq[1])
    # print(fft_freq == top3_freq[2])

    # fft_1x[fft_freq != peak_freq] = 0
    # filtered_motion = 2 * np.fft.ifft(fft_1x)
    # cycle = round(1 / frame_duration / peak_freq)

    # plt.plot(filtered_motion)
    # plt.plot(ref_jnt_pos[:, 1])
    # plt.show()


if __name__ == "__main__":
    main()

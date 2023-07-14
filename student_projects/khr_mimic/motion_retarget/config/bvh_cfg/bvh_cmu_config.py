import numpy as np

# -----------------------------------------SFU Skeleton Configuration-----------------------
FILE_DIR = "bvh_data/CMU/009/"
# file names list which should be retargeted
# 54_01, 54_14, 54_08, 90_19, 90_20, 90_21
FILE_LIST = ["09_01.bvh"]
SAVE_TEMP_FILE = True
TEMP_FILE_DIR = "result/temp/"
OUT_FILE_DIR = "result/output/"
# scale the global position of all joints during calculating global reference position for joints cm--->m
MOTION_WEIGHT = 1.0
POSITION_SCALING = 0.01
Y_UP_AXIS = True  # the coordinate used in bvh file
LEFT_HAND_COORDINATE = False
FRAME_DURATION = 0.008333

BVH_JOINT_NAMES = [
    "Hips",
    "LHipJoint",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "RHipJoint",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "LowerBack",
    "Spine",
    "Spine1",
    "Neck",
    "Neck1",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftFingerBase",
    "LeftHandIndex1",
    "LThumb",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightFingerBase",
    "RightHandIndex1",
    "RThumb",
]

BVH_ROOT_HEIGHT = 0.157 + 0.154 + 0.015
BVH_CHESET_LENGTH = 0.103 + 0.078
BVH_HAND_LENGTH = 0.1094 + 0.0852

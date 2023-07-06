# KHR Deep Mimic 

Project for KHR robot locomotion with Deep Mimic.


![](asset/khr_mimic.gif)
## Prepare
Tested on Ubuntu 20.04 (probably no problem with 18.04 and MacOS cause they support mujoco and mujoco-py).

### Setup
- Install and setup Mujoco210 and mujoco-py environment following this [link](https://gist.githubusercontent.com/saratrajput/60b1310fe9d9df664f9983b38b50d5da/raw/4b3ab6af7de1f336f75fbfa0d93116e8d5d1bbc4/mujoco_py_install_instructions.md).
- Then, install python requirements below.
    ```bash
    pip install -r requirements.txt
    ```
### Run
- Train
    ```bash
    python3 scripts/train.py --n_env 8 --et --control_type torque
    ```
- Run pretrained policy (trained for 6 hours) 
    ```bash
    python3 scripts/train.py --et --control_type torque --what test --trial 17
    ```

## TODO
- Tidy up train and sim code
- Upload motion retargeting and processing code (using pybullet)
- More diverse motion

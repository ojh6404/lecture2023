
import os
import re
import time
import glob
import argparse
import numpy as np
import pandas as pd
import torch

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed, get_device, get_latest_run_id
from stable_baselines3.common.callbacks import CheckpointCallback
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from ramiel_sim import RamielEnv

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--what", type=str, default="train", help="what do you want to do: [train, test]")
    parser.add_argument("--render", type=int, default=1, help="Render or not (only when testing)")
    parser.add_argument("--trial", type=int, default=None, help="PPO trial number to load")
    parser.add_argument("--iter", type=int, default=None, help="PPO trial number to load. if None, best iter is loaded")
    parser.add_argument("--max_step", type=int, default=20000000, help="PPO train total timesteps")
    parser.add_argument("--n_step", type=int, default=2048, help="number of steps for each env to update")
    parser.add_argument("--batch", type=int, default=64, help="number of batch to update")
    parser.add_argument("--epoch", type=int, default=10, help="number of epoch to update")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--save_interval", type=int, default=10, help="interval of iteration to save network weight")
    parser.add_argument("--n_env", type=int, default=1, help="number of env to use")
    parser.add_argument("--wait", action="store_true", help="wait of render when testing")
    parser.add_argument('--resume', action="store_true", help='resume the training')
    parser.add_argument('--ros', action="store_true", help='publish some info using ros when testing')
    parser.add_argument("--best_rate", type=float, default=0.0, help="if 0.0, choose best snapshot from all iterations")
    return parser

def make_env(seed, max_step, rank=0):
    def _init(test=False, ros=False):
        env = Monitor(RamielEnv(test, ros, max_step))
        env.seed(seed + rank)
        return env
    return _init

def main():
    args = parser().parse_args()
    set_random_seed(args.seed)

    if args.what == "train":
        env = SubprocVecEnv([make_env(args.seed, int(args.max_step/args.n_env), i) for i in range(args.n_env)])
    else:
        env = make_env(args.seed, None)(test=True, ros=args.ros)


    root_dir = os.path.dirname(os.path.abspath(__file__))
    model = PPO("MlpPolicy",
            env,
            policy_kwargs = dict(
                activation_fn=torch.nn.Tanh,
                net_arch=dict(pi=[256, 128], vf=[256, 128]),
                log_std_init=-0.5,
                ),
            n_steps = args.n_step,
            batch_size = args.batch,
            n_epochs = args.epoch,
            gamma = args.gamma,
            verbose=1,
            tensorboard_log=root_dir+"/../saved")

    policy = None
    if (args.what == "test") or args.resume:
        if args.trial is not None:
            load_iter = None
            if args.iter is not None:
                load_iter = args.iter
            else: # best iter is extracted
                tlog_path = glob.glob(root_dir + "/../saved/PPO_{0}/events.out*".format(args.trial))[0]
                tlog = EventAccumulator(tlog_path)
                tlog.Reload()
                rew_data = pd.DataFrame(tlog.Scalars('rollout/ep_rew_mean'))
                model_list = sorted(glob.glob(root_dir + "/../saved/Policy_{0}/*".format(args.trial)), key=lambda x: int(re.findall(r'\d+', x)[-1]))
                n_all_iter = len(rew_data.iloc[args.save_interval-1::args.save_interval])
                sorted_rew_data = (rew_data.iloc[args.save_interval-1::args.save_interval])[int(args.best_rate*n_all_iter):].sort_values(by="value")
                print(sorted_rew_data)
                load_iter = sorted_rew_data.tail(1).index[0]
                load_step = sorted_rew_data.loc[load_iter, 'step']
            weight = root_dir + "/../saved/Policy_{0}/model_{1}_steps".format(args.trial, load_step)
            print("load: {}".format(weight))
            model = model.load(weight, print_system_info=True)

    if args.what == "train":
        trial = get_latest_run_id(root_dir + "/../saved", "PPO") + 1
        save_freq = args.n_step*args.save_interval
        checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=root_dir + "/../saved/Policy_{0}/".format(trial), name_prefix='model')
        model.learn(total_timesteps=args.max_step, callback=checkpoint_callback)
    elif args.what == "test":
        step = 0
        start_time = time.time()
        obs = env.reset()
        for i in range(args.max_step):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            step += 1
            if args.render:
                env.render()
            if done:
                print("step: {}".format(step))
                print("----------")
                obs = env.reset()
                step = 0

            end_time = time.time()
            # print("{} [sec]".format(end_time-start_time))
            if float(end_time-start_time) < env.dt:
                if args.wait:
                    time.sleep(env.dt-float(end_time-start_time))
            start_time = end_time
    else:
        import ipdb
        ipdb.set_trace()

if __name__ == '__main__':
    main()


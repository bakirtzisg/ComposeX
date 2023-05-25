import gym
import robosuite_wrapper

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

env_name = 'Box'  # 'Move' or 'Place'

# Parallel environments
vec_env = make_vec_env(f'{env_name}-v1', n_envs=4)

model = PPO('MlpPolicy', vec_env, verbose=1)
model.learn(total_timesteps=25000)
model.save(f'ppo_{env_name}')

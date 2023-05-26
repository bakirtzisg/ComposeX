import gym
import robosuite_wrapper

from stable_baselines3 import SAC

env_name = 'Box'  # 'Move' or 'Place'

env = gym.make(f'{env_name}-v1')

model = SAC('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=50000, log_interval=4)
model.save(f'sac_{env_name}')

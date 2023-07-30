import gymnasium as gym 
import compx
import numpy as np
from stable_baselines3 import SAC


env = gym.make('CompPickPlaceCan-v1')
sac_agents = {}
for task in env.unwrapped.tasks:
    sac_agents[task] = SAC.load(f'sac_{task}')

done = True

while True:
    if done:
        obs, info = env.reset()
    obs = env.unwrapped._get_obs()
    current_task = env.unwrapped.current_task
    action, _states = sac_agents[current_task].predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()

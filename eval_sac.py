import gym
import robosuite_wrapper

from stable_baselines3 import SAC

env_name = 'Box'  # 'Move' or 'Place'

env = gym.make(f'{env_name}-v1')

model = SAC.load(f'sac_{env_name}_10000')

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

import gym
import robosuite_wrapper

from stable_baselines3 import SAC

env_name = 'Lift'  # 'Move' or 'Place'

env = gym.make(f'{env_name}-v1')

model = SAC.load(f'sac_{env_name}_500000')

obs = env.reset()
episode_reward = 0
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    episode_reward += reward
    env.render()
    if done:
        obs = env.reset()
        print('Episode reward:', episode_reward)
        episode_reward = 0

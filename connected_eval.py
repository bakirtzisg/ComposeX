import gym
import robosuite_wrapper
from stable_baselines3 import SAC

env = gym.make(f'Can-v1')

model = SAC.load('sac_ReachAbove_500000')

obs = env.reset()
episode_reward = 0
current_sub_mdp = 'reach_above'
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    episode_reward += reward
    env.render()
    if done:
        obs = env.reset()
        print('Episode reward:', episode_reward)
        episode_reward = 0
    if env.unwrapped.sub_mdp != current_sub_mdp:
        current_sub_mdp = env.unwrapped.sub_mdp
        print(f"Sub-MDP changed to {current_sub_mdp}")
        if current_sub_mdp == 'lift':
            model = SAC.load('sac_Lift_500000')
        elif current_sub_mdp == 'move':
            model = SAC.load('sac_Move')
        elif current_sub_mdp == 'place':
            model = SAC.load('sac_Place')

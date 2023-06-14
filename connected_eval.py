import gym
import robosuite_wrapper

from stable_baselines3 import SAC

env = gym.make('Complete-v1')

obs = env.reset()
sub_mdp = 'box'
model = SAC.load('sac_box')
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        if sub_mdp == 'box':
            env.unwrapped.sub_mdp = 'move'
            model = SAC.load('sac_move')
            sub_mdp = 'move'
        elif sub_mdp == 'move':
            env.unwrapped.sub_mdp = 'place'
            model = SAC.load('sac_place')
            sub_mdp = 'place'
        else:
            obs = env.reset()
            sub_mdp = 'box'
            model = SAC.load('sac_box')

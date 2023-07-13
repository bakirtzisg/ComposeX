import gym
import robosuite_wrapper
import numpy as np
from stable_baselines3 import SAC

env_name = 'Lift'  # 'Move' or 'Place'

print(env_name)
env = gym.make(f'{env_name}-v1')

model = SAC('MlpPolicy', env, verbose=1,
            learning_rate=0.001, tensorboard_log=f"./tb/{env_name}/")
# model = SAC.load(f'sac_{env_name}_50000', env=env,
#                  learning_rate=0.001,
#                  verbose=1)


def collect_expert_data(num_episodes=10):
    for _ in range(num_episodes):
        obs = env.reset()
        done = False

        phase = 0
        while not done:
            if env_name == 'Box':
                if phase == 0:
                    if np.linalg.norm(obs[3:6] + np.array([0, 0, 0.1]) - obs[:3]) < 0.01:
                        phase = 1
                    action = -np.ones(3, dtype=np.float32)
                    action[:3] = ((obs[3:6] + np.array([0, 0, 0.1]) - obs[:3]) * 10).clip(-1, 1)

            if env_name == 'Lift':
                if phase == 0:
                    if np.linalg.norm(obs[1:2] + np.array([0.005]) - obs[0:1]) < 0.01:
                        phase = 1
                    action = -np.ones(2, dtype=np.float32)
                    action[0] = ((obs[1] + 0.005 - obs[0]) * 10).clip(-1, 1)
                if phase == 1:
                    if obs[2] < 0.055:
                        phase = 2
                    action = np.zeros(2, dtype=np.float32)
                    action[1] = 1
                if phase == 2:
                    action = np.ones(2, dtype=np.float32)

            if env_name == 'Place':
                if phase == 0:
                    if np.linalg.norm(obs[6:9] + np.array([0, 0, 0.2]) - obs[:3]) < 0.1:
                        phase = 1
                    action = np.ones(4, dtype=np.float32)
                    action[:3] = ((obs[6:9] + np.array([0, 0, 0.1]) - obs[:3]) * 10).clip(-1, 1)
                if phase == 1:
                    action = np.zeros(4, dtype=np.float32)
                    action[3] = -1

            next_obs, reward, done, info = env.step(action)
            env.render()
            model.replay_buffer.add(obs, next_obs, action, reward, done, [info])
            obs = next_obs
            if done:
                print('Final step reward:', reward)

if env_name in ['Box', 'Place', 'Lift']:
    collect_expert_data(10)
model.learn(total_timesteps=500000, log_interval=4)
model.save(f'sac_{env_name}_500000')

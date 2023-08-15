import compx
import gymnasium as gym
import numpy as np

e = gym.make('BaselineCompNut-v1')
obs, info = e.reset()

stage = 1
wait = 0

while True:
    e.render()

    if stage == 1:
        action = np.zeros(4)
        goal = obs[3:6]
        goal[1] += 0.04
        action[:3] = ((goal - obs[0:3]) * 10).clip(-1, 1)
        if np.linalg.norm(action) < 0.01:
            stage = 2
        print(obs[:3])  # 0.835 grip
    if stage == 2:
        action = np.zeros(4)
        action[3] = 1
        wait += 1
        if wait > 3:
            stage = 3
            wait = 0
    if stage == 3:
        action = np.zeros(4)
        goal = obs[6:9]
        goal[2] = 1
        action[:3] = ((goal - obs[3:6]) * 10).clip(-1, 1)
        # if np.linalg.norm(action) < 0.01:
        #     stage = 4
        wait += 1
        if wait > 20:
            stage = 4
    if stage == 4:
        action = np.zeros(4)
        action[3] = -1
        
    obs, reward, terminated, truncated, info = e.step(action)
    done = terminated or truncated

    if done:
        obs, info = e.reset()
        stage = 1
        wait = 0

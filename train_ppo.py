import gym
import robosuite_wrapper

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env("Box-v1", n_envs=4)
# vec_env = make_vec_env("Move-v1", n_envs=4)
# vec_env = make_vec_env("Place-v1", n_envs=4)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo_box")

import gymnasium as gym 
import compx
import numpy as np
from stable_baselines3 import SAC


TOTAL_TRAINING_STEPS = 1000000
LOG_INTERVAL = 4

env = gym.make('CompPickPlaceCan-v1')
sac_agents = {}
for task in env.unwrapped.tasks:
    env.unwrapped.current_task = task
    sac_agents[task] = SAC('MlpPolicy', env, verbose=1,
                           gamma=0.96,
                           tensorboard_log=f'./tb/{task}/')
env.unwrapped.current_task = env.unwrapped.tasks[0]

setup_done = {}
for task in env.unwrapped.tasks:
    setup_done[task] = False

total_timesteps, callback = sac_agents['reach_above']._setup_learn(
    total_timesteps=TOTAL_TRAINING_STEPS,
    callback=None,
    reset_num_timesteps=True,
    tb_log_name='sac',
    progress_bar=False,
)
setup_done['reach_above'] = True

for step in range(TOTAL_TRAINING_STEPS):
    task = env.unwrapped.current_task
    task_sac = sac_agents[task]

    if not setup_done[task]:
        env.unwrapped._setup_skip_reset()
        total_timesteps, callback = task_sac._setup_learn(
            total_timesteps=TOTAL_TRAINING_STEPS,
            callback=None,
            reset_num_timesteps=True,
            tb_log_name='sac',
            progress_bar=False,
        )
        setup_done[task] = True

    task_sac.policy.set_training_mode(False)

    if env.unwrapped.fresh_reset:
        task_sac._last_obs = env.unwrapped._get_obs()[None]
    actions, buffer_actions = task_sac._sample_action(task_sac.learning_starts, task_sac.action_noise, task_sac.env.num_envs)

    new_obs, rewards, dones, infos = task_sac.env.step(actions)

    rewards[0] = infos[0]['task_reward']
    dones[0] = infos[0]['task_terminated'] or dones[0]

    task_sac.num_timesteps += task_sac.env.num_envs

    # Retrieve reward and episode length if using Monitor wrapper
    task_sac._update_info_buffer(infos, dones)

    # Store data in replay buffer (normalized action and unnormalized observation)
    task_sac._store_transition(task_sac.replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

    task_sac._update_current_progress_remaining(task_sac.num_timesteps, task_sac._total_timesteps)

    for idx, done in enumerate(dones):
        if done:
            task_sac._episode_num += 1

            # Log training infos
            if LOG_INTERVAL is not None and task_sac._episode_num % LOG_INTERVAL == 0:
                task_sac._dump_logs()

    task_sac.train(batch_size=task_sac.batch_size, gradient_steps=1)

for task in env.unwrapped.tasks:
    sac_agents[task].save(f'sac_{task}')


# model = SAC('MlpPolicy', env, verbose=1,
#             learning_rate=0.001, tensorboard_log=f"./tb/{env_name}/")
# model = SAC.load(f'sac_{env_name}_50000', env=env,
#                  learning_rate=0.001,
#                  verbose=1)


# if env_name in ['Box', 'Place', 'Lift']:
#     collect_expert_data(10)
# model.learn(total_timesteps=500000, log_interval=4)
# model.save(f'sac_{env_name}_500000')

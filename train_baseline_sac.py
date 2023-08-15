import gymnasium as gym 
import compx
import numpy as np
import time
import argparse
from stable_baselines3 import SAC


TOTAL_TRAINING_STEPS = 1000000
LOG_INTERVAL = 4
EVAL_INTERVAL = 2000
NUM_EVAL_EPISODES = 20


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env_name', type=str, default='BaselineCompLift-v1')
    # parser.add_argument('--env_name', type=str, default='BaselineCompStack-v1')
    parser.add_argument('--env_name', type=str, default='BaselineCompPickPlaceCan-v1')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    experiment_path = f'./experiments/{args.env_name}/{time.strftime("%Y%m%d-%H%M%S")}-id-{np.random.randint(10000)}/'
    env = gym.make(args.env_name)

    sac_agent = SAC('MlpPolicy', env, verbose=1, gamma=0.96, tensorboard_log=experiment_path + f'tb/')

    eval_env = gym.make(args.env_name)
    def evaluate():
        successes = np.zeros(NUM_EVAL_EPISODES, dtype=bool)
        episode_rewards = np.zeros(NUM_EVAL_EPISODES)
        episode_lengths = np.zeros(NUM_EVAL_EPISODES)
        for i in range(NUM_EVAL_EPISODES):
            obs, info = eval_env.reset()
            done = False
            length = 0
            while not done:
                action, _states = sac_agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                episode_rewards[i] += info['task_reward']
                length += 1
            if terminated and 'episode_success' in info and info['episode_success']:
                successes[i] = True
            episode_lengths[i] = length
        return successes, episode_rewards, episode_lengths
    eval_results = {}

    total_timesteps, callback = sac_agent._setup_learn(
        total_timesteps=TOTAL_TRAINING_STEPS,
        callback=None,
        reset_num_timesteps=True,
        tb_log_name='sac',
        progress_bar=False,
    )

    print(f'Evaluating at step 0...')
    successes, episode_rewards, episode_lengths = evaluate()
    eval_results[0] = {
        'successes': successes,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }
    print(eval_results[0])

    for step in range(TOTAL_TRAINING_STEPS):
        sac_agent.policy.set_training_mode(False)
        actions, buffer_actions = sac_agent._sample_action(sac_agent.learning_starts, sac_agent.action_noise, sac_agent.env.num_envs)

        new_obs, rewards, dones, infos = sac_agent.env.step(actions)

        sac_agent.num_timesteps += sac_agent.env.num_envs

        # Retrieve reward and episode length if using Monitor wrapper
        sac_agent._update_info_buffer(infos, dones)

        # Store data in replay buffer (normalized action and unnormalized observation)
        sac_agent._store_transition(sac_agent.replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

        sac_agent._update_current_progress_remaining(sac_agent.num_timesteps, sac_agent._total_timesteps)

        for idx, done in enumerate(dones):
            if done:
                sac_agent._episode_num += 1

                # Log training infos
                if LOG_INTERVAL is not None and sac_agent._episode_num % LOG_INTERVAL == 0:
                    sac_agent._dump_logs()

        sac_agent.train(batch_size=sac_agent.batch_size, gradient_steps=1)

        if (step + 1) % EVAL_INTERVAL == 0:
            print(f'Evaluating at step {step}...')
            successes, episode_rewards, episode_lengths = evaluate()
            eval_results[step + 1] = {
                'successes': successes,
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths
            }
            print(eval_results[step + 1])

    sac_agent.save(experiment_path + f'sac_agent')
    np.save(experiment_path + 'eval_results.npy', eval_results)

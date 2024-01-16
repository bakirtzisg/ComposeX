import gymnasium as gym
import compx
import numpy as np
import time
import argparse
from stable_baselines3 import SAC


TOTAL_TRAINING_STEPS = 500000
LOG_INTERVAL = 4
EVAL_INTERVAL = 2000
NUM_EVAL_EPISODES = 20


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="CompLift-v1")
    # parser.add_argument("--env_name", type=str, default="CompStack-v1")
    # parser.add_argument("--env_name", type=str, default="CompNut-v1")
    # parser.add_argument("--env_name", type=str, default="CompPickPlaceCan-v1")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    experiment_path = f'./experiments/{args.env_name}/{time.strftime("%Y%m%d-%H%M%S")}-id-{np.random.randint(10000)}/'
    env = gym.make(args.env_name)

    # Initialize SAC agents for each sub-task. The environment is returned to the first sub-task.
    sac_agents = {}
    setup_done = {}
    for task in env.unwrapped.tasks:
        env.unwrapped.current_task = task
        sac_agents[task] = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            gamma=0.96,
            tensorboard_log=experiment_path + f"tb/{task}/",
        )
        setup_done[task] = False
    env.unwrapped.current_task = env.unwrapped.tasks[0]

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
                obs = eval_env.unwrapped._get_obs()
                current_task = eval_env.unwrapped.current_task
                action, _states = sac_agents[current_task].predict(
                    obs, deterministic=True
                )
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                episode_rewards[i] += info["task_reward"]
                length += 1
            if terminated and "episode_success" in info and info["episode_success"]:
                successes[i] = True
            episode_lengths[i] = length
        return successes, episode_rewards, episode_lengths

    eval_results = {}

    first_task = env.unwrapped.current_task
    total_timesteps, callback = sac_agents[first_task]._setup_learn(
        total_timesteps=TOTAL_TRAINING_STEPS,
        callback=None,
        reset_num_timesteps=True,
        tb_log_name="sac",
        progress_bar=False,
    )
    setup_done[first_task] = True

    print(f"Evaluating at step 0...")
    successes, episode_rewards, episode_lengths = evaluate()
    eval_results[0] = {
        "successes": successes,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }
    print(eval_results[0])

    for step in range(TOTAL_TRAINING_STEPS):
        task = env.unwrapped.current_task
        task_sac = sac_agents[task]

        if not setup_done[task]:
            env.unwrapped._setup_skip_reset()
            total_timesteps, callback = task_sac._setup_learn(
                total_timesteps=TOTAL_TRAINING_STEPS,
                callback=None,
                reset_num_timesteps=True,
                tb_log_name="sac",
                progress_bar=False,
            )
            setup_done[task] = True

        task_sac.policy.set_training_mode(False)

        if env.unwrapped.fresh_reset:
            task_sac._last_obs = env.unwrapped._get_obs()[None]

            # Manully reset the Montor environment
            task_sac.env.envs[0].rewards = []

        actions, buffer_actions = task_sac._sample_action(
            task_sac.learning_starts, task_sac.action_noise, task_sac.env.num_envs
        )
        new_obs, rewards, dones, infos = task_sac.env.step(actions)

        # When the entire task is done, logging has been automatically done.
        already_logged = dones[0]

        # Set dones for sub-tasks.
        dones[0] = infos[0]["task_terminated"] or dones[0]

        task_sac.num_timesteps += task_sac.env.num_envs

        # Log rollout info in the Monitor environment
        if dones[0] and not already_logged:
            monitor_env = task_sac.env.envs[0]
            if not monitor_env.needs_reset:
                ep_rew = sum(monitor_env.rewards)
                ep_len = len(monitor_env.rewards)
                ep_info = {
                    "r": round(ep_rew, 6),
                    "l": ep_len,
                    "t": round(time.time() - monitor_env.t_start, 6),
                }
                for key in monitor_env.info_keywords:
                    ep_info[key] = infos[0][key]
                monitor_env.episode_returns.append(ep_rew)
                monitor_env.episode_lengths.append(ep_len)
                monitor_env.episode_times.append(time.time() - monitor_env.t_start)
                ep_info.update(monitor_env.current_reset_info)
                if monitor_env.results_writer:
                    monitor_env.results_writer.write_row(ep_info)
                infos[0]["episode"] = ep_info
        # Retrieve reward and episode length if using Monitor wrapper
        task_sac._update_info_buffer(infos, dones)

        # Store data in replay buffer (normalized action and unnormalized observation)
        task_sac._store_transition(
            task_sac.replay_buffer, buffer_actions, new_obs, rewards, dones, infos
        )

        task_sac._update_current_progress_remaining(
            task_sac.num_timesteps, task_sac._total_timesteps
        )

        for idx, done in enumerate(dones):
            if done:
                task_sac._episode_num += 1

                # Log training infos
                if (
                    LOG_INTERVAL is not None
                    and task_sac._episode_num % LOG_INTERVAL == 0
                ):
                    task_sac._dump_logs()

        task_sac.train(batch_size=task_sac.batch_size, gradient_steps=1)
        task_sac.logger.record("train/global_step", step)

        if (step + 1) % EVAL_INTERVAL == 0:
            print(f"Evaluating at step {step}...")
            successes, episode_rewards, episode_lengths = evaluate()
            eval_results[step + 1] = {
                "successes": successes,
                "episode_rewards": episode_rewards,
                "episode_lengths": episode_lengths,
            }
            print(eval_results[step + 1])

    for task in env.unwrapped.tasks:
        sac_agents[task].save(experiment_path + f"sac_{task}")
    np.save(experiment_path + "eval_results.npy", eval_results)

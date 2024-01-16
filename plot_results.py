import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(font_scale=1.5)

PLOT_METRIC = "Success Rate"
# PLOT_METRIC = 'Mean Episode Reward'
ENV_NAME = "Nut"
TITLE = "Assemble a Nut"
PLOT_SKIP = 2

# experiments = [f'BaselineComp{ENV_NAME}-v1', f'Comp{ENV_NAME}-v1', f'Comp{ENV_NAME}-v1-recycle']
# labels = ['Baseline', 'Ours', 'Ours (Skill Finetune)']
experiments = [f"BaselineComp{ENV_NAME}-v1", f"Comp{ENV_NAME}-v1"]
labels = ["Baseline", "Ours"]


def compute_metrics(file_path):
    eval_results = np.load(file_path, allow_pickle=True).item()
    steps = list(eval_results.keys())
    steps.sort()
    success_rates = np.empty(len(steps))
    mean_episode_rewards = np.empty(len(steps))
    for i, step in enumerate(steps):
        success_rates[i] = np.mean(eval_results[step]["successes"])
        mean_episode_rewards[i] = np.mean(eval_results[step]["episode_rewards"])
    return steps, success_rates, mean_episode_rewards


def smooth(y, box_pts):
    y = np.concatenate([[y[0]] * (box_pts // 2), y, [y[-1]] * (box_pts // 2)])
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="valid")
    return y_smooth


def compute_mean_std(metric_arrays):
    metric_arrays = np.array(metric_arrays)
    mean = np.mean(metric_arrays, axis=0)
    std = np.std(metric_arrays, axis=0)
    return mean, std


for i in range(len(experiments)):
    experiment = experiments[i]
    label = labels[i]
    base_path = f"./results/{experiment}"
    folders = os.listdir(base_path)
    success_rates = []
    mean_episode_rewards = []
    for folder in folders:
        file_path = f"{base_path}/{folder}/eval_results.npy"
        steps, success_rate, mean_episode_reward = compute_metrics(file_path)
        steps = steps[::PLOT_SKIP]
        success_rate = success_rate[::PLOT_SKIP]
        mean_episode_reward = mean_episode_reward[::PLOT_SKIP]
        success_rate = smooth(success_rate, 5)
        success_rates.append(success_rate)
        mean_episode_reward = smooth(mean_episode_reward, 5)
        mean_episode_rewards.append(mean_episode_reward)

    if PLOT_METRIC == "Success Rate":
        mean, std = compute_mean_std(success_rates)
        plt.plot(steps, mean, label=label)
        plt.fill_between(
            steps, (mean - std).clip(0, 1), (mean + std).clip(0, 1), alpha=0.2
        )
    elif PLOT_METRIC == "Mean Episode Reward":
        mean, std = compute_mean_std(mean_episode_rewards)
        plt.plot(steps, mean, label=label)
        plt.fill_between(steps, mean - std, mean + std, alpha=0.2)

plt.legend()
plt.xlabel("Environment Steps")
plt.ylabel(PLOT_METRIC)
plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
plt.title(TITLE)
plt.tight_layout()
# plt.show()
plt.savefig(f"{ENV_NAME}-{PLOT_METRIC}.pdf")

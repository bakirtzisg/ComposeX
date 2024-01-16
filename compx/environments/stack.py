import gymnasium as gym
import robosuite
import numpy as np
from robosuite import load_controller_config
from gymnasium.envs.registration import register


class CompStackEnv(gym.Env):
    def __init__(self, baseline_mode=False):
        config = load_controller_config(default_controller='OSC_POSITION')
        self._env = robosuite.make(
            env_name="Stack",
            robots="Panda",
            controller_configs=config,
            has_renderer=True,
            has_offscreen_renderer=False,
            ignore_done=False,
            use_camera_obs=False,
            control_freq=10,
            horizon=200
        )
        self.tasks = ['reach', 'lift', 'place']
        self.action_spaces = {
            'reach': gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
            'lift': gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            'place': gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        }
        self.observation_spaces = {
            'reach': gym.spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32),
            'lift': gym.spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32),
            'place': gym.spaces.Box(low=-10, high=10, shape=(10,), dtype=np.float32)
        }

        self.current_task = 'reach'
        self.reward_criteria = None
        self.setup_skip_reset_once = False
        self.fresh_reset = False
        self.baseline_mode = baseline_mode

    @property
    def action_space(self):
        if self.baseline_mode:
            return gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        else:
            return self.action_spaces[self.current_task]
    
    @property
    def observation_space(self):
        if self.baseline_mode:
            return gym.spaces.Box(low=-10, high=10, shape=(10,), dtype=np.float32)
        else:
            return self.observation_spaces[self.current_task]

    def _get_obs(self):
        obs = self._env._get_observations()

        hand_pos = obs['robot0_eef_pos']
        cubeA_pos = obs['cubeA_pos']
        cubeB_pos = obs['cubeB_pos']
        gripper = obs['robot0_gripper_qpos'][0] * 2

        if self.baseline_mode:
            obs = np.concatenate([hand_pos, cubeA_pos, cubeB_pos, [gripper]]).astype(np.float32)
            return obs

        if self.current_task == 'reach':
            obs = np.concatenate([hand_pos, cubeA_pos]).astype(np.float32)
        elif self.current_task == 'lift':
            obs = np.concatenate([hand_pos[2:3], cubeA_pos[2:3], [gripper]]).astype(np.float32)
        elif self.current_task == 'place':
            obs = np.concatenate([hand_pos, cubeA_pos, cubeB_pos, [gripper]]).astype(np.float32)
        else:
            raise RuntimeError('Invalid task')
        return obs

    def _process_action(self, action):
        if self.baseline_mode:
            return action

        if self.current_task == 'lift':
            action = np.concatenate([[0, 0], action])
        if self.current_task == 'reach':
            action = np.concatenate([action, [-1]])
        return action
    
    def _compute_reward_criteria(self, observation):
        hand_pos = observation['robot0_eef_pos'].copy()
        cubeA_pos = observation['cubeA_pos'].copy()
        goal_pos = observation['cubeB_pos'].copy()
        goal_pos[2] = 0.9
        reach_dist = np.linalg.norm(cubeA_pos - hand_pos)
        goal_dist = np.linalg.norm(goal_pos - cubeA_pos)
        criteria = {
            'reach_dist': reach_dist,
            'goal_dist': goal_dist,
            'cubeA_height': cubeA_pos[2]
        }
        return criteria

    def _evaluate_task(self, observation):
        task_completed = False
        task_failed = False

        new_reward_criteria = self._compute_reward_criteria(observation)
        if self.baseline_mode:
            task_reward = self.reward_criteria['reach_dist'] - new_reward_criteria['reach_dist']
            task_reward += new_reward_criteria['cubeA_height'] - self.reward_criteria['cubeA_height']
            task_reward += self.reward_criteria['goal_dist'] - new_reward_criteria['goal_dist']
        else:
            if self.current_task == 'reach':
                task_reward = self.reward_criteria['reach_dist'] - new_reward_criteria['reach_dist']
                if new_reward_criteria['reach_dist'] < 0.01:
                    task_completed = True
            elif self.current_task == 'lift':
                task_reward = self.reward_criteria['reach_dist'] - new_reward_criteria['reach_dist']
                task_reward += new_reward_criteria['cubeA_height'] - self.reward_criteria['cubeA_height']
                if new_reward_criteria['cubeA_height'] > 0.84:
                    task_completed = True
            elif self.current_task == 'place':
                task_reward = self.reward_criteria['goal_dist'] - new_reward_criteria['goal_dist']

            if task_completed:
                task_reward = 10

        self.reward_criteria = new_reward_criteria
        return task_reward, task_completed, task_failed

    def step(self, action):
        self.fresh_reset = False
        action = self._process_action(action)
        observation, reward, done, info = self._env.step(action)
        task_reward, task_completed, task_failed = self._evaluate_task(observation)

        terminated = done
        truncated = False
        # The entire task is completed if the can is placed in the bin.
        if reward > 0:
            info['episode_success'] = True
            terminated = True
            task_reward = 10
        if task_failed:
            terminated = True

        info['task_reward'] = task_reward
        info['task_terminated'] = task_completed or task_failed or terminated

        if task_completed:
            previous_task_obs = self._get_obs()
            self.current_task = self.tasks[self.tasks.index(self.current_task) + 1]
            info['current_task'] = self.current_task
            info['current_task_obs'] = self._get_obs()
            obs = previous_task_obs
            self.fresh_reset = True
        else:
            obs = self._get_obs()
        return obs, task_reward, terminated, truncated, info

    def _setup_skip_reset(self):
        self.setup_skip_reset_once = True

    def reset(self, seed=None, options=None):
        if self.setup_skip_reset_once:
            self.setup_skip_reset_once = False
            obs = self._get_obs()
        else:
            if self.fresh_reset:
                print(f'Called reset right after task switch to {self.current_task}.')
                if self.current_task != 'reach':
                    self.current_task = self.tasks[self.tasks.index(self.current_task) - 1]
            observation = self._env.reset()
            obs = self._get_obs()
            self.current_task = 'reach'
            self.reward_criteria = self._compute_reward_criteria(observation)
        self.fresh_reset = True
        return obs, {'current_task': self.current_task,
                     'current_task_obs': self._get_obs()}

    def render(self, *args, **kwargs):
        self._env.render()

    def close(self) -> None:
        self._env.close()
        return super().close()


def register_envs():
    register(
        id="CompStack-v1",
        entry_point=CompStackEnv,
        max_episode_steps=75
    )
    register(
        id="BaselineCompStack-v1",
        entry_point=CompStackEnv,
        max_episode_steps=75,
        kwargs={'baseline_mode': True}
    )

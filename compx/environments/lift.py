import gymnasium as gym
import robosuite
import numpy as np
from robosuite import load_controller_config
from gymnasium.envs.registration import register

class CompLiftEnv(gym.Env):
    def __init__(self):
        config = load_controller_config(default_controller='OSC_POSITION')
        self._env = robosuite.make(
            env_name="Lift",
            robots="Panda",
            controller_configs=config,
            has_renderer=True,
            has_offscreen_renderer=False,
            ignore_done=False,
            use_camera_obs=False,
            control_freq=10,
            horizon=100
        )
        self.tasks = ['reach', 'lift']
        self.action_spaces = {
            'reach': gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
            'lift': gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        }
        self.observation_spaces = {
            'reach': gym.spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32),
            'lift': gym.spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32),
        }

        self.current_task = 'reach'
        self.init_cube_pos = None
        self.reward_criteria = None
        self.setup_skip_reset_once = False
        self.fresh_reset = False

    @property
    def action_space(self):
        return self.action_spaces[self.current_task]
    
    @property
    def observation_space(self):
        return self.observation_spaces[self.current_task]

    def _get_obs(self):
        obs = self._env._get_observations()

        hand_pos = obs['robot0_eef_pos']
        cube_pos = obs['cube_pos']
        gripper = obs['robot0_gripper_qpos'][0] * 2

        if self.current_task == 'reach':
            obs = np.concatenate([hand_pos, cube_pos]).astype(np.float32)
        elif self.current_task == 'lift':
            obs = np.concatenate([hand_pos[2:3], cube_pos[2:3], [gripper]]).astype(np.float32)
        else:
            raise RuntimeError('Invalid task')
        return obs
    
    def _get_hand_pos(self):
        obs = self._env._get_observations()
        return obs['robot0_eef_pos']
    
    def _get_cube_pos(self):
        obs = self._env._get_observations()
        return obs['cube_pos']

    def _clip_hand_pos(self, hand_pos):
        return hand_pos

    def _process_action(self, action):
        if self.current_task == 'lift':
            action = np.concatenate([[0, 0], action])
        hand_pos = self._get_hand_pos()
        resulting_hand_pos = hand_pos + action[:3] / 10
        cliped_hand_pos = self._clip_hand_pos(resulting_hand_pos)
        action[:3] = ((cliped_hand_pos - hand_pos) * 10).clip(-1, 1)
        if self.current_task == 'reach':
            action = np.concatenate([action, [-1]])
        return action
    
    def _compute_reward_criteria(self, observation):
        hand_pos = observation['robot0_eef_pos'].copy()
        cube_pos = observation['cube_pos'].copy()
        reach_dist = np.linalg.norm(cube_pos - hand_pos)
        criteria = {
            'reach_dist': reach_dist,
            'cube_height': cube_pos[2]
        }
        return criteria

    def _evaluate_task(self, observation):
        task_completed = False
        task_failed = False

        new_reward_criteria = self._compute_reward_criteria(observation)
        if self.current_task == 'reach':
            task_reward = self.reward_criteria['reach_dist'] - new_reward_criteria['reach_dist']
            if new_reward_criteria['reach_dist'] < 0.01:
                task_completed = True
        elif self.current_task == 'lift':
            task_reward = self.reward_criteria['reach_dist'] - new_reward_criteria['reach_dist']
            task_reward += new_reward_criteria['cube_height'] - self.reward_criteria['cube_height']
            # if observation['cube_pos'][2] > 1:
            #     task_completed = True

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
            self.init_cube_pos = self._get_cube_pos()
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
        id="CompLift-v1",
        entry_point=CompLiftEnv,
        max_episode_steps=50
    )

import gymnasium as gym
import robosuite
import numpy as np
from robosuite import load_controller_config
from gymnasium.envs.registration import register

# odict_keys(['robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 'robot0_eef_pos',
# 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'agentview_image', 'cubeA_pos',
# 'cubeA_quat', 'cubeB_pos', 'cubeB_quat', 'gripper_to_cubeA',
# 'gripper_to_cubeB', 'cubeA_to_cubeB', 'robot0_proprio-state', 'object-state'])

class CompPickPlaceCanEnv(gym.Env):
    def __init__(self):
        config = load_controller_config(default_controller='OSC_POSITION')
        self._env = robosuite.make(
            env_name="PickPlaceCan",
            robots="Panda",
            controller_configs=config,
            has_renderer=True,
            has_offscreen_renderer=False,
            ignore_done=False,
            use_camera_obs=False,
            control_freq=10,
            horizon=200
        )
        self.tasks = ['reach', 'lift', 'move', 'place']
        self.action_spaces = {
            'reach': gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
            'lift': gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            'move': gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
            'place': gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        }
        self.observation_spaces = {
            'reach': gym.spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32),
            'lift': gym.spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32),
            'move': gym.spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32),
            'place': gym.spaces.Box(low=-10, high=10, shape=(10,), dtype=np.float32)
        }

        self.current_task = 'reach'
        self.init_can_pos = None
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
        can_pos = obs['Can_pos']
        gripper = obs['robot0_gripper_qpos'][0] * 2
        goal_pos = self._env.target_bin_placements[self._env.object_to_id['can']]

        if self.current_task == 'reach':
            obs = np.concatenate([hand_pos, can_pos]).astype(np.float32)
        elif self.current_task == 'lift':
            obs = np.concatenate([hand_pos[2:3], can_pos[2:3], [gripper]]).astype(np.float32)
        elif self.current_task == 'move':
            obs = hand_pos.astype(np.float32)
        elif self.current_task == 'place':
            obs = np.concatenate([hand_pos, can_pos, goal_pos, [gripper]]).astype(np.float32)
        else:
            raise RuntimeError('Invalid task')
        return obs
    
    def _get_hand_pos(self):
        obs = self._env._get_observations()
        return obs['robot0_eef_pos']
    
    def _get_can_pos(self):
        obs = self._env._get_observations()
        return obs['Can_pos']

    def _clip_hand_pos(self, hand_pos):
        if self.current_task == 'reach':
            hand_pos = hand_pos.clip([-0.15, -0.5, 0.8], [0.35, 0, 1.1])
        elif self.current_task == 'move':
            hand_pos = hand_pos.clip([-0.15, -0.5, 1], [0.35, 0.53, 1.2])
        elif self.current_task == 'place':
            hand_pos = hand_pos.clip([-0.15, 0.03, 1], [0.35, 0.53, 1.2])
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
        elif self.current_task == 'move':
            action = np.concatenate([action, [1]])
        return action
    
    def _compute_reward_criteria(self, observation):
        hand_pos = observation['robot0_eef_pos'].copy()
        can_pos = observation['Can_pos'].copy()
        reach_dist = np.linalg.norm(can_pos + np.array([0, 0, 0.005]) - hand_pos)
        move_dist = 0.03 - hand_pos[1]
        goal_dist = np.linalg.norm(can_pos - self._env.target_bin_placements[self._env.object_to_id['can']])
        can_displacement = np.linalg.norm(can_pos - self.init_can_pos)
        criteria = {
            'reach_dist': reach_dist,
            'move_dist': move_dist,
            'goal_dist': goal_dist,
            'can_displacement': can_displacement,
            'can_height': can_pos[2]
        }
        return criteria

    def _evaluate_task(self, observation):
        task_completed = False
        task_failed = False

        new_reward_criteria = self._compute_reward_criteria(observation)
        if self.current_task == 'reach':
            task_reward = self.reward_criteria['reach_dist'] - new_reward_criteria['reach_dist']
            # task_reward = -reach_dist * np.exp(reach_dist)
            if new_reward_criteria['can_displacement'] > 0.01:
                task_failed = True
            elif new_reward_criteria['reach_dist'] < 0.01:
                task_completed = True
        elif self.current_task == 'lift':
            task_reward = self.reward_criteria['reach_dist'] - new_reward_criteria['reach_dist']
            task_reward += new_reward_criteria['can_height'] - self.reward_criteria['can_height']
            # task_reward = -grasp_dist * np.exp(grasp_dist)
            if observation['Can_pos'][2] > 1:
                task_completed = True
        elif self.current_task == 'move':
            task_reward = self.reward_criteria['move_dist'] - new_reward_criteria['move_dist']
            # task_reward = -move_dist * np.exp(move_dist)
            if observation['robot0_eef_pos'][1] > 0.03:
                task_completed = True
        elif self.current_task == 'place':
            task_reward = self.reward_criteria['goal_dist'] - new_reward_criteria['goal_dist']
            # task_reward = -goal_dist * np.exp(goal_dist)

        # task_reward = abs(task_reward)

        if task_completed:
            task_reward = 10
        # elif task_failed:
        #     task_reward = -100

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
            self.init_can_pos = self._get_can_pos()
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
        id="CompPickPlaceCan-v1",
        entry_point=CompPickPlaceCanEnv,
        max_episode_steps=100
    )

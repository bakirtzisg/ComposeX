import gym
import robosuite
import numpy as np
from robosuite import load_controller_config
from gym.envs.registration import register

class RobosuiteWrapper(gym.Env):
    def __init__(self, horizon=100, sub_mdp=None):
        self.camera_names = ['frontview', 'robot0_eye_in_hand']

        config = load_controller_config(default_controller='OSC_POSITION')

        # PickAndPlace env
        # bin1_pos=(0.1, -0.25, 0.8)
        # bin2_pos=(0.1, 0.28, 0.8)
        env = robosuite.make(
            env_name="PickPlaceCan",
            robots="Panda",
            controller_configs=config,
            has_renderer=True,
            has_offscreen_renderer=False,
            ignore_done=False,
            use_camera_obs=False,
            control_freq=10,
            horizon=1000
        )
        self.horizon = horizon
        self._env = env
        self.sub_mdp = sub_mdp

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=float)
        # self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(7,), dtype=float)

    def _get_obs(self):
        obs = self._env._get_observations()
        return self._unpack_obs(obs)
    
    def move_to_pos(self, goal_pos, gripper_action=0, tol=0.01, render=False):
        obs = self._get_obs()
        hand_pos = obs['hand_pos']
        while np.linalg.norm(goal_pos - hand_pos) > tol:
            action = np.zeros(4)
            action[:3] = (goal_pos - hand_pos) * 10
            action[3] = gripper_action
            obs, _, _, _ = self.step(action)
            hand_pos = obs['hand_pos']
            if render:
                self._env.render()

    def fast_forward(self, mode='box', render=False):
        if mode == 'box':
            self.move_to_pos(np.array([0.1, -0.25, 1]), gripper_action=-1, render=render)

        # holding the can
        if mode == 'move':
            obs = self._get_obs()
            can_pos = obs['can_pos']
            self.move_to_pos(can_pos + np.array([0, 0, 0.1]), gripper_action=-1, render=render)
            self.move_to_pos(can_pos, gripper_action=-1, render=render)
            for _ in range(4):
                self.step(np.array([0, 0, 0, 1]))
                if render:
                    self._env.render()
        
        if mode == 'place':
            self.fast_forward(mode='move', render=render)
            obs = self._get_obs()
            hand_pos = obs['hand_pos']
            raised_pos = hand_pos.copy()
            raised_pos[2] = 1
            self.move_to_pos(raised_pos, gripper_action=1, render=render)

    def _unpack_obs(self, obs):
        unpacked_obs = {
            'hand_pos': obs['robot0_eef_pos'],
            'can_pos': obs['Can_pos'],
            'gripper': abs(obs['robot0_gripper_qpos'][0]) * 2
        }
        return unpacked_obs

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = self._unpack_obs(obs)
        return obs, reward, done, info

    def reset(self):
        self._env.reset()
        if self.sub_mdp is not None:
            self.fast_forward(mode=self.sub_mdp)
        return self._get_obs()

    def render(self, *args, **kwargs):
        self._env.render(*args, **kwargs)

    @property
    def _max_episode_steps(self):
        return self.horizon

    @property
    def observation_space(self):
        return self._observation_space
    
    def close(self) -> None:
        self._env.close()
        return super().close()


def register_envs():
    register(
        id="Box-v1",
        entry_point=RobosuiteWrapper,
        max_episode_steps=100,
        kwargs={'sub_mdp': 'box'}
    )
    register(
        id="Move-v1",
        entry_point=RobosuiteWrapper,
        max_episode_steps=100,
        kwargs={'sub_mdp': 'move'}
    )
    register(
        id="Place-v1",
        entry_point=RobosuiteWrapper,
        max_episode_steps=100,
        kwargs={'sub_mdp': 'place'}
    )


register_envs()


if __name__ == '__main__':
    env = RobosuiteWrapper()
    env.reset()
    env.fast_forward('place', render=True)

import gym
import robosuite
import numpy as np
from robosuite import load_controller_config
from gym.envs.registration import register

class RobosuiteWrapper(gym.Env):
    def __init__(self, horizon=100, fast_forward=None):
        self.camera_names = ['frontview', 'robot0_eye_in_hand']

        config = load_controller_config(default_controller='OSC_POSITION')

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
        self.fast_forward = fast_forward
        self._env = env
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=float)

    def _get_obs(self):
        obs = self._env._get_observations()
        return self._unpack_obs(obs)
    
    def heuritic_fast_forward(self, mode='reach', render=False):
        if mode == 'reach':  # move to the can
            obs = self._get_obs()
            can_pos = obs['can_pos']
            hand_pos = obs['hand_pos']
            while np.linalg.norm(can_pos + np.array([0, 0, 0.1]) - hand_pos) > 0.01:
                heuristic_action = -np.ones(4)
                heuristic_action[:3] = (can_pos + np.array([0, 0, 0.1]) - hand_pos) * 10
                obs, _, _, _ = self.step(heuristic_action)
                can_pos = obs['can_pos']
                hand_pos = obs['hand_pos']
                if render:
                    self._env.render()
            while np.linalg.norm(can_pos + np.array([0, 0, 0.005]) - hand_pos) > 0.01:
                heuristic_action = -np.ones(4)
                heuristic_action[:3] = (can_pos + np.array([0, 0, 0.005]) - hand_pos) * 10
                obs, _, _, _ = self.step(heuristic_action)
                can_pos = obs['can_pos']
                hand_pos = obs['hand_pos']
                if render:
                    self._env.render()
        if mode == 'grasp':
            self.heuritic_fast_forward(mode='reach')
            for _ in range(4):
                self.step(np.array([0, 0, 0, 1]))
                if render:
                    self._env.render()

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
        if self.fast_forward is not None:
            self.heuritic_fast_forward(mode=self.fast_forward)
        return self._get_obs()

    def render(self, *args, **kwargs):
        self._env.render()

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
        id="ReachCan-v1",
        entry_point=RobosuiteWrapper,
        max_episode_steps=100,
        kwargs={'fast_forward': None}
    )
    register(
        id="GraspCan-v1",
        entry_point=RobosuiteWrapper,
        max_episode_steps=100,
        kwargs={'fast_forward': 'reach'}
    )
    register(
        id="PlaceCan-v1",
        entry_point=RobosuiteWrapper,
        max_episode_steps=100,
        kwargs={'fast_forward': 'grasp'}
    )


register_envs()


if __name__ == '__main__':
    env = RobosuiteWrapper('PickPlaceCan')
    o = env.reset()
    env.heuritic_fast_forward('grasp', render=True)

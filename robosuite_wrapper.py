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
            has_renderer=False,
            has_offscreen_renderer=False,
            ignore_done=False,
            use_camera_obs=False,
            control_freq=10,
            horizon=1000
        )
        self.horizon = horizon
        self._env = env
        self.sub_mdp = sub_mdp

        # self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=float)
        # self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(7,), dtype=float)

    @property
    def action_space(self):
        if self.sub_mdp == 'move':
            return gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        else:
            return gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=float)
    
    @property
    def observation_space(self):
        if self.sub_mdp == 'box':
            return gym.spaces.Box(low=-10, high=10, shape=(7,), dtype=float)
        elif self.sub_mdp == 'move':
            return gym.spaces.Box(low=-10, high=10, shape=(3,), dtype=float)
        elif self.sub_mdp == 'place':
            return gym.spaces.Box(low=-10, high=10, shape=(7,), dtype=float)
        else:
            raise NotImplementedError

    def _get_obs(self):
        obs = self._env._get_observations()
        return self._process_obs(obs)
    
    def _get_hand_pos(self):
        obs = self._env._get_observations()
        return obs['robot0_eef_pos']
    
    def _get_can_pos(self):
        obs = self._env._get_observations()
        return obs['Can_pos']

    def move_to_pos(self, goal_pos, gripper_action=0, tol=0.01, render=False):
        hand_pos = self._get_hand_pos()
        while np.linalg.norm(goal_pos - hand_pos) > tol:
            action = np.zeros(4)
            action[:3] = (goal_pos - hand_pos) * 10
            action[3] = gripper_action
            obs, _, _, _ = self._env.step(action)
            hand_pos = obs['robot0_eef_pos']
            if render:
                self._env.render()

    def fast_forward(self, mode='box', render=False):
        if mode == 'box':
            self.move_to_pos(np.array([0.1, -0.25, 1]), gripper_action=-1, render=render)

        if mode == 'move':
            # Grab the can
            can_pos = self._get_can_pos()
            self.move_to_pos(can_pos + np.array([0, 0, 0.1]), gripper_action=-1, render=render)
            self.move_to_pos(can_pos, gripper_action=-1, render=render)
            for _ in range(4):
                self._env.step(np.array([0, 0, 0, 1]))
                if render:
                    self._env.render()
            
            # Lift it off the box
            hand_pos = self._get_hand_pos()
            raised_pos = hand_pos.copy()
            raised_pos[2] = 1
            self.move_to_pos(raised_pos, gripper_action=1, render=render)
        
        if mode == 'place':
            self.fast_forward(mode='move', render=render)
            place_boundary_pos = np.random.uniform([-0.15, 0.03, 1], [0.35, 0.03, 1.2])
            self.move_to_pos(place_boundary_pos, gripper_action=1, render=render)

    def _process_obs(self, obs):
        hand_pos = obs['robot0_eef_pos']
        can_pos = obs['Can_pos']
        gripper = abs(obs['robot0_gripper_qpos'][0]) * 2
        bin = self._env.target_bin_placements
        bin_id = self._env.object_to_id['can']
        goal_pos = bin[bin_id]

        if self.sub_mdp == 'box':
            obs = np.concatenate([hand_pos, can_pos, [gripper]])
        elif self.sub_mdp == 'move':
            obs = hand_pos
        elif self.sub_mdp == 'place':
            obs = np.concatenate([hand_pos, goal_pos, [gripper]])
        else:
            obs = np.concatenate([hand_pos, can_pos, [gripper]])

        return obs
    
    def _clip_hand_pos(self, hand_pos):
        if self.sub_mdp == 'box':
            hand_pos = hand_pos.clip([-0.15, -0.5, 0.8], [0.35, 0, 1.1])
        elif self.sub_mdp == 'move':
            hand_pos = hand_pos.clip([-0.15, -0.5, 1], [0.35, 0.53, 1.2])
        elif self.sub_mdp == 'place':
            hand_pos = hand_pos.clip([-0.15, 0.03, 1], [0.35, 0.53, 1.2])
        return hand_pos

    def _advance_sub_mdp(self):
        if self.sub_mdp == 'box':
            if np.linalg.norm(self._get_hand_pos() - self._get_can_pos()) < 0.01 and self._get_hand_pos()[2] > 1:
                return True
                # self.sub_mdp = 'move'
        elif self.sub_mdp == 'move':
            if self._get_hand_pos()[1] > 0.03:
                return True
                # self.sub_mdp = 'place'
        return False

    def _process_action(self, action):
        hand_pos = self._get_hand_pos()
        resulting_hand_pos = hand_pos + action[:3] / 10
        cliped_hand_pos = self._clip_hand_pos(resulting_hand_pos)
        action[:3] = ((cliped_hand_pos - hand_pos) * 10).clip(-1, 1)
        if self.sub_mdp == 'move':
            action = np.concatenate([action, [1]])
        return action

    def step(self, action):
        action = self._process_action(action)
        obs, reward, done, info = self._env.step(action)
        sub_mdp_changed = self._advance_sub_mdp()
        obs = self._process_obs(obs)
        if sub_mdp_changed or reward > 0:
            reward = 1
            done = True
        else:
            reward = 0
            done = False
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
    input()

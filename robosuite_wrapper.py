import gym
import robosuite
import numpy as np
from robosuite import load_controller_config
from gym.envs.registration import register

class RobosuiteWrapper(gym.Env):
    def __init__(self, horizon=100, set_sub_mdp=None):
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
        if set_sub_mdp is not None:
            self.sub_mdp = set_sub_mdp
            self.set_done_sub_mdp = True
        else:
            self.sub_mdp = 'reach_above'
            self.set_done_sub_mdp = False

        self.init_can_pos = None

    @property
    def action_space(self):
        if self.sub_mdp in ['reach_above', 'move']:
            return gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        elif self.sub_mdp == 'lift':
            return gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        else:
            return gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
    
    @property
    def observation_space(self):
        if self.sub_mdp == 'reach_above':
            return gym.spaces.Box(low=-10, high=10, shape=(7,), dtype=np.float32)
        elif self.sub_mdp == 'lift':
            return gym.spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32)
        elif self.sub_mdp == 'move':
            return gym.spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32)
        elif self.sub_mdp == 'place':
            return gym.spaces.Box(low=-10, high=10, shape=(10,), dtype=np.float32)
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

    def fast_forward(self, mode='reach_above', render=False):
        if mode == 'reach_above':
            self.move_to_pos(np.array([0.1, -0.25, 1]), gripper_action=-1, render=render)
        
        if mode == 'lift':
            can_pos = self._get_can_pos()
            self.move_to_pos(can_pos + np.array([0, 0, 0.1]))

        if mode == 'move':
            # Grab the can
            can_pos = self._get_can_pos()
            self.move_to_pos(can_pos + np.array([0, 0, 0.1]), gripper_action=-1, render=render)
            self.move_to_pos(can_pos + np.array([0, 0, 0.01]), gripper_action=-1, render=render)
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

        if self.sub_mdp == 'reach_above':
            obs = np.concatenate([hand_pos, can_pos, [gripper]])
        elif self.sub_mdp == 'lift':
            obs = np.concatenate([hand_pos[2:3], can_pos[2:3], [gripper]])
        elif self.sub_mdp == 'move':
            obs = hand_pos
        elif self.sub_mdp == 'place':
            obs = np.concatenate([hand_pos, can_pos, goal_pos, [gripper]])
        else:
            obs = np.concatenate([hand_pos, can_pos, [gripper]])

        return obs
    
    def _clip_hand_pos(self, hand_pos):
        if self.sub_mdp == 'reach_above':
            hand_pos = hand_pos.clip([-0.15, -0.5, 0.8], [0.35, 0, 1.1])
        elif self.sub_mdp == 'move':
            hand_pos = hand_pos.clip([-0.15, -0.5, 1], [0.35, 0.53, 1.2])
        elif self.sub_mdp == 'place':
            hand_pos = hand_pos.clip([-0.15, 0.03, 1], [0.35, 0.53, 1.2])
        return hand_pos

    def _advance_sub_mdp(self):
        if self.sub_mdp == 'reach_above':
            hand_pos = self._get_hand_pos()
            can_pos = self._get_can_pos()
            above_dist = np.linalg.norm(can_pos + np.array([0, 0, 0.1]) - hand_pos)
            if above_dist < 0.01:
                self.sub_mdp = 'lift'
                if self.set_done_sub_mdp:
                    return True
        if self.sub_mdp == 'lift':
            if self._get_can_pos()[2] > 1:
                self.sub_mdp = 'move'
                if self.set_done_sub_mdp:
                    return True
        elif self.sub_mdp == 'move':
            if self._get_hand_pos()[1] > 0.03:
                self.sub_mdp = 'place'
                if self.set_done_sub_mdp:
                    return True
        return False

    def _process_action(self, action):
        if self.sub_mdp == 'lift':
            action = np.concatenate([[0, 0], action])
        hand_pos = self._get_hand_pos()
        resulting_hand_pos = hand_pos + action[:3] / 10
        cliped_hand_pos = self._clip_hand_pos(resulting_hand_pos)
        action[:3] = ((cliped_hand_pos - hand_pos) * 10).clip(-1, 1)
        if self.sub_mdp == 'reach_above':
            action = np.concatenate([action, [-1]])
        elif self.sub_mdp == 'move':
            action = np.concatenate([action, [1]])
        return action

    def step(self, action):
        action = self._process_action(action)
        
        obs, reward, done, info = self._env.step(action)
        done = self._advance_sub_mdp()

        hand_pos = obs['robot0_eef_pos'].copy()
        can_pos = obs['Can_pos'].copy()
        can_dist = np.linalg.norm(can_pos + np.array([0, 0, 0.005]) - hand_pos)
        goal_dist = np.linalg.norm(can_pos - self._env.target_bin_placements[self._env.object_to_id['can']])

        obs = self._process_obs(obs)

        if reward > 0:
            done = True

        if self.sub_mdp == 'reach_above':
            # reward = -can_dist + max(can_pos[2] - 0.86, 0) - 1
            above_dist = np.linalg.norm(can_pos + np.array([0, 0, 0.1]) - hand_pos)
            reward = -above_dist * np.exp(above_dist)

            can_move = np.linalg.norm(can_pos - self.init_can_pos)
            if can_move > 0.01:
                reward = -100
                done = True
            elif above_dist < 0.01:
                reward = 100
                done = True
            
            # r_reach, r_grasp, r_lift, r_hover = self._env.staged_rewards()
            # if r_grasp > 0:
            #     done = True
            # reward = r_reach
        elif self.sub_mdp == 'lift':
            reward = -can_dist * np.exp(can_dist)
            if done:
                reward = 100

        elif self.sub_mdp == 'move':
            reward = hand_pos[1]
        elif self.sub_mdp == 'place':
            reward = -goal_dist

        # if done:
        #     reward = 100

        return obs, reward, done, info

    def reset(self):
        self._env.reset()
        if self.sub_mdp is not None:
            try:
                self.fast_forward(mode=self.sub_mdp)
            except ValueError:
                self.reset()
        if self.sub_mdp == 'reach_above':
            self.init_can_pos = self._get_can_pos()
        return self._get_obs()

    def render(self, *args, **kwargs):
        self._env.render()

    @property
    def _max_episode_steps(self):
        return self.horizon

    def close(self) -> None:
        self._env.close()
        return super().close()


def register_envs():
    register(
        id="ReachAbove-v1",
        entry_point=RobosuiteWrapper,
        max_episode_steps=100,
        kwargs={'set_sub_mdp': 'reach_above'}
    )
    register(
        id="Lift-v1",
        entry_point=RobosuiteWrapper,
        max_episode_steps=100,
        kwargs={'set_sub_mdp': 'lift'}
    )
    register(
        id="Move-v1",
        entry_point=RobosuiteWrapper,
        max_episode_steps=100,
        kwargs={'set_sub_mdp': 'move'}
    )
    register(
        id="Place-v1",
        entry_point=RobosuiteWrapper,
        max_episode_steps=100,
        kwargs={'set_sub_mdp': 'place'}
    )
    register(
        id="Can-v1",
        entry_point=RobosuiteWrapper,
        max_episode_steps=100,
        kwargs={'set_sub_mdp': None}
    )


register_envs()


if __name__ == '__main__':
    env = RobosuiteWrapper(sub_mdp='move')
    env.reset()
    # env.fast_forward('place', render=True)
    input()

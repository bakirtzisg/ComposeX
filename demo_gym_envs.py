import gym
import robosuite_wrapper
import time


SLEEP_TIME = 1  # seconds


input("Env 1: Reach the can. Press Enter to continue...")
e = gym.make('ReachCan-v1')
e.reset()
e.render()
time.sleep(SLEEP_TIME)
e.close()

input("Env 2: Grasp the can. Press Enter to continue...")
e = gym.make('GraspCan-v1')
e.reset()
e.render()
time.sleep(SLEEP_TIME)
e.close()

input("Env 3: Place the can. Press Enter to continue...")
e = gym.make('PlaceCan-v1')
e.reset()
e.render()
time.sleep(SLEEP_TIME)
e.close()

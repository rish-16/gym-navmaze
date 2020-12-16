import gym
import gym_maze
import numpy as np
import matplotlib.pyplot as plt

from gym_maze.envs import MazeEnv
from gym_maze.envs.generators import TMazeGenerator, RandomBlockMazeGenerator
from gym_maze.envs.Astar_solver import AstarSolver

import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env

from mem_buff import MemBuff

# maze = TMazeGenerator(3, [5, 3], [3, 3])
maze = RandomBlockMazeGenerator(maze_size=30, obstacle_ratio=0.2)
env = MazeEnv(maze, action_type="Moore", render_trace=False)
walls = env.reset()

obs_dim = walls.shape
n_actions = env.action_space.n
print (obs_dim, n_actions)

solver = AstarSolver(env, env.goal_states[0])

mem_buff = MemBuff(obs_dim, n_actions)

if not solver.solvable():
    raise ValueError('The maze is not solvable given the current state and the goal state')

ideal_solution = solver.get_actions()
print (ideal_solution)

for action in solver.get_actions():
    walls, reward, done, info = env.step(action)
    
# env = DummyVecEnv([lambda: env])

# model = PPO2(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=25000)
# model.save("ppo2_navmaze")

# del model # remove to demonstrate saving and loading

# model = PPO2.load("ppo2_navmaze")

# # Enjoy trained agent
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     walls, rewards, dones, info = env.step(action)

def env_creator(env_config):
    return env
    
ray.init()    
register_env("navmaze", env_creator)    
    
trainer = PPOTrainer(env="navmaze")

while True:
    print (trainer.train())


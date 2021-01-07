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

maze = RandomBlockMazeGenerator(maze_size=20, obstacle_ratio=0.2)
env = MazeEnv(maze, action_type="VonNeumann", render_trace=False)
walls = env.reset()

obs_dim = walls.shape
n_actions = env.action_space.n
print (obs_dim, n_actions)

solver = AstarSolver(env, env.goal_states[0])

if not solver.solvable():
    raise ValueError('The maze is not solvable given the current state and the goal state')

ideal_solution = solver.get_actions()

# for action in solver.get_actions():
    # walls, reward, done, info = env.step(action)
    
ray.init()    
register_env("navmaze", lambda env_config : env)
    
trainer = PPOTrainer(env="navmaze")

# while True:
    # print (trainer.train())
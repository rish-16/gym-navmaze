import gym
import gym_maze
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

from gym_maze.envs import MazeEnv
from gym_maze.envs.generators import TMazeGenerator, RandomBlockMazeGenerator
from gym_maze.envs.Astar_solver import AstarSolver

import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env

maze = RandomBlockMazeGenerator(maze_size=14, obstacle_ratio=0.2)
env = MazeEnv(maze, size=14, action_type="VonNeumann", render_trace=False)

num_moves_lookup = [[np.inf for j in range(15)] for i in range(15)]

print (env.goal_states[0])
for s in env.free_spaces:
    env.init_state = s
    solver = AstarSolver(env, env.goal_states[0])
    if solver.solvable():
        ideal = solver.get_actions()
        num_moves_lookup[env.init_state[0]][env.init_state[1]] = len(ideal)
    
    env.reset()
    
env.lookup_table = num_moves_lookup

print (env.lookup_table)
    
ray.init()
register_env("navmaze", lambda env_config : env)

trainer = PPOTrainer(env="navmaze")

while True:
    print (trainer.train())
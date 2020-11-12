import gym
import gym_maze
import numpy as np

from gym_maze.envs import MazeEnv
from gym_maze.envs.generators import TMazeGenerator
from gym_maze.envs.Astar_solver import AstarSolver

from mem_buff import MemBuff

maze = TMazeGenerator(3, [5, 3], [3, 3])
env = MazeEnv(maze, action_type="VonNeumann", render_trace=False)
obs = env.reset()

obs_dim = obs.reshape([21, 24, 1]).shape
n_actions = env.action_space.n

solver = AstarSolver(env, env.goal_states[0])

mem_buff = MemBuff(obs_dim, n_actions)

if not solver.solvable():
    raise ValueError('The maze is not solvable given the current state and the goal state')

ideal_solution = solver.get_actions()

for action in solver.get_actions():
    obs, reward, done, info = env.step(action)
    env.render()
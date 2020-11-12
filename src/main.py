import gym
import gym_maze
import numpy as np

from gym_maze.envs import MazeEnv
from gym_maze.envs.generators import TMazeGenerator

maze = TMazeGenerator(3, [5, 3], [3, 3])
env = MazeEnv(maze, action_type="VonNeumann", render_trace=False)

obs = env.reset()

while True:
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    
    if done:
        break
    
env.close()    
import logging
import gym, torch, gym_maze
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

from gym_maze.envs import MazeEnv
from gym_maze.envs.generators import TMazeGenerator, RandomBlockMazeGenerator
from gym_maze.envs.Astar_solver import AstarSolver

from methods import PPO

def display(mp):
    for i in range(len(mp)):
        header = ""
        for j in range(len(mp[i])):
            header += "{:^5}".format(mp[i][j])
        print (header)

maze = RandomBlockMazeGenerator(maze_size=14, obstacle_ratio=0.2)
env = MazeEnv(maze, size=14, action_type="VonNeumann", render_trace=False)
env.seed(1)
num_moves_lookup = [[np.inf for j in range(15)] for i in range(15)]
T_horizon = 20

print (env.goal_states[0])

for s in env.free_spaces:
    env.init_state = s
    env.reset()
    solver = AstarSolver(env, env.goal_states[0])
    
    if solver.solvable():
        ideal = solver.get_actions()
        num_moves_lookup[env.init_state[0]][env.init_state[1]] = len(ideal)
    
    env.reset()
    
env.lookup_table = num_moves_lookup
display(num_moves_lookup)

learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20

model = PPO()
score = 0.0
print_interval = 20

all_frames = []

for n_epi in range(10000):
    s, frame = env.reset()
    all_frames.append(frame)
    done = False

    while not done:
        for t in range(T_horizon):
            prob = model.pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample().item()
            s_prime, r, done, info = env.step(a)
            all_frames.append(info['frame'])

            model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done)) 
            s = s_prime

            score += r

            if done:
                break

        model.train_net()

    if n_epi % print_interval == 0 and n_epi != 0:
        print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
        score = 0.0

print (len(all_frames))

env.close()
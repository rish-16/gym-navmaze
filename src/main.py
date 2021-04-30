import logging
import gym, torch, gym_maze
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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

MAZE_SIZE = 7
maze = RandomBlockMazeGenerator(maze_size=MAZE_SIZE-1, obstacle_ratio=0.2)
env = MazeEnv(maze, size=14, action_type="VonNeumann", render_trace=False)
env.seed(1)
num_moves_lookup = [[np.inf for j in range(MAZE_SIZE)] for i in range(MAZE_SIZE)]

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

T_horizon = 40

model = PPO(6, 4)
score = 0.0
print_interval = 2

all_frames = []
traces = []

for n_epi in range(100):
    s, frame = env.reset()
    done = False

    print (n_epi)

    episodic_succ_rate = 0
    episodic_trace = []
    episodic_frames = [frame]

    t = 0
    while not done and t < T_horizon:
        env.render()
        prob = model.pi(torch.from_numpy(s).flatten().float())
        m = Categorical(prob)
        a = m.sample().item()
        print (a, m)
        s_prime, r, done, info = env.step(a)
        
        # log action-frame history
        episodic_trace.append(a)
        episodic_frames.append(info['frame'])

        model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done)) 
        s = s_prime

        score += r
        t += 1

        if done:
            episodic_succ_rate += 1
            break

        model.train_net()

    print ("# of episodes: {}, avg. score: {:.1f}".format(n_epi, score/print_interval))
    score = 0.0

print (len(all_frames))
env.close()
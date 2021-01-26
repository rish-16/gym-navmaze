import gym, torch, gym_maze
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

from gym_maze.envs import MazeEnv
from gym_maze.envs.generators import TMazeGenerator, RandomBlockMazeGenerator
from gym_maze.envs.Astar_solver import AstarSolver

import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from methods import PPO_LSTM

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

ray.init()
register_env("navmaze", lambda env_config : env)

config = ppo.DEFAULT_CONFIG.copy()
config["model"]["use_lstm"] = True

trainer = PPOTrainer(env="navmaze", config=config)

while True:
    print (trainer.train())

# model = PPO_LSTM(env.observation_space.shape[0], env.action_space.n)
# score = 0

# rewards = []
# success_rate = 0

# for ep in range(1000000):
#     h_out = (torch.zeros([1, 1, 32], dtype=torch.float), torch.zeros([1, 1, 32], dtype=torch.float))
#     s = env.reset()
#     done = False
    
#     while not done:
#         for t in range(T_horizon):
#             h_in = h_out
#             prob, h_out = model.pi(torch.from_numpy(s).float(), h_in)
#             prob = prob.view(-1)
#             m = Categorical(prob)
#             action = m.sample().item()
#             s_prime, r, done, info = env.step(action)
#             rewards.append(r)
            
#             model.put_data((s, action, r, s_prime, prob[action].item(), h_in, h_out, done))
#             s = s_prime
            
#             score += r
#             if done:
#                 done = True
#                 success_rate += 1
#                 break
                
#         model.train()
        
#     if (ep % 20 == 0 and ep != 0):
#         print ("# of episode : {} | avg score : {:.1f}".format(ep, score/20))
#         score = 0.0
        
# env.close()
# print (success_rate)
# plt.plot(list(range(1000000)), rewards)
# plt.show()
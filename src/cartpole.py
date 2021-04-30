import numpy as np
from collections import deque
import pickle
import torch
from utils import collect_trajectories, random_sample
from PPO import PPO
import matplotlib.pyplot as plt
from parallelEnv import *
import gym

env = gym.make("CartPole-v0")
env.reset()
env.seed(2)

obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
act_dist = [0 for i in range(n_actions)]

def train(episode, env_name):
    gamma = .99
    gae_lambda = 0.95
    use_gae = True
    beta = .01
    cliprange = 0.1
    best_score = -np.inf
    goal_score = 195.0
    ep_length = []

    nenvs = 1
    rollout_length = 200
    minibatches = 10*8
    nbatch = nenvs * rollout_length
    optimization_epochs = 4
    
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    envs = parallelEnv(env_name, nenvs, seed=1234)
    agent = PPO(state_size=obs_dim,
                     action_size=n_actions,
                     seed=0,
                     hidden_layers=[64,64],
                     lr_policy=1e-4, 
                     use_reset=True,
                     device=device)

    print(agent.policy)

    # keep track of progress
    mean_rewards = []
    scores_window = deque(maxlen=100)
    loss_storage = []

    for i_episode in range(episode+1):
        log_probs_old, states, actions, rewards, values, dones, vals_last, infos, ep_length = collect_trajectories(envs, act_dist, ep_length, agent.policy, rollout_length)

        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        
        if not use_gae:
            for t in reversed(range(rollout_length)):
                if t == rollout_length - 1:
                    returns[t] = rewards[t] + gamma * (1-dones[t]) * vals_last
                else:
                    returns[t] = rewards[t] + gamma * (1-dones[t]) * returns[t+1]
                advantages[t] = returns[t] - values[t]
        else:
            for t in reversed(range(rollout_length)):
                if t == rollout_length - 1:
                    returns[t] = rewards[t] + gamma * (1-dones[t]) * vals_last
                    td_error = returns[t] - values[t]
                else:
                    returns[t] = rewards[t] + gamma * (1-dones[t]) * returns[t+1]
                    td_error = rewards[t] + gamma * (1-dones[t]) * values[t+1] - values[t]
                advantages[t] = advantages[t] * gae_lambda * gamma * (1-dones[t]) + td_error
        
        # convert to pytorch tensors and move to gpu if available
        returns = torch.from_numpy(returns).float().to(device).view(-1,)
        advantages = torch.from_numpy(advantages).float().to(device).view(-1,)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        
        for _ in range(optimization_epochs):
            sampler = random_sample(nbatch, minibatches)
            for inds in sampler:
                mb_log_probs_old = log_probs_old[inds]
                mb_states = states[inds]
                mb_actions = actions[inds]
                mb_returns = returns[inds]
                mb_advantages = advantages[inds]
                loss_p, loss_v, loss_ent = agent.update(mb_log_probs_old, mb_states, mb_actions, mb_returns, mb_advantages, cliprange=cliprange, beta=beta)
                loss_storage.append([loss_p, loss_v, loss_ent])
                
        total_rewards = np.sum(rewards, axis=0)
        scores_window.append(np.mean(total_rewards)) # last 100 scores
        mean_rewards.append(np.mean(total_rewards))  # get the average reward of the parallel environments
        cliprange *= 0.999                              # the clipping parameter reduces as time goes on
        beta *= 0.999                                   # the regulation term reduces
    
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            print(total_rewards)
        if np.mean(scores_window)>=goal_score and np.mean(scores_window)>=best_score:            
            torch.save(agent.policy.state_dict(), "policy_cartpole.pth")
            best_score = np.mean(scores_window)
    
    return mean_rewards, loss_storage, act_dist, ep_length

mean_rewards, loss, new_act_dist, ep_length = train(10000, 'CartPole-v0')

print (new_act_dist[-1])
print (ep_length)

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.size'] = 10

plt.title("PPO + MLP + GAE for 10000 episodes")

plt.subplot(131)
plt.plot(mean_rewards)
plt.ylabel('Average score')
plt.xlabel('Episode')

plt.subplot(132)
plt.plot(list(range(len(ep_length))), ep_length, color="red")
plt.ylabel('Episode Length')
plt.xlabel('Episode')

plt.subplot(133)
plt.ylabel('Frequency')
plt.xlabel('Actions')
plt.bar(['Action {}'.format(i) for i in range(len(new_act_dist))], new_act_dist[-1])

plt.show()
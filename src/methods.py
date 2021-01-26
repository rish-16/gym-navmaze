import numpy as np
import torch, time
from torch import nn
from torch.nn import functional as F
from torch.utils import tensorboard
import torch.optim as optim

class PPO_LSTM(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.data = []
        
        self.fc1 = nn.Linear(obs_dim, 64)
        self.lstm = nn.LSTM(64, 32)
        self.fc_pi = nn.Linear(32, n_actions)
        self.fc_v = nn.Linear(32, 1)
        
        self.gamma = 0.98
        self.lmbda = 0.95
        self.alpha = 0.0005
        self.eps_clip = 0.1
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)
        
    def pi(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 64)
        x, lstm_hidden = self.lstm(x, hidden)
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=2)

        return prob, lstm_hidden
        
    def v(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 64)
        x, lstm_hid = self.lstm(x, hidden)
        v = self.fc_v(x)
        return v
        
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, h_in_lst, h_out_lst, done_lst = [], [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_pr, prob_a, h_in, h_out, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_pr)
            prob_a_lst.append([prob_a])
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
            s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float),torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
            
            self.data = [] # reset data for new batch
            
            return s, a, r, s_prime, done_mask, prob_a, h_in_lst[0], h_out_lst[0]
            
    def train(self):
        s, a, r, s_prime, done_mask, prob_a, (h1_in, h2_in), (h1_out, h2_out) = self.make_batch()
        first_hidden  = (h1_in.detach(), h2_in.detach())
        second_hidden = (h1_out.detach(), h2_out.detach())

        for i in range(2):
            v_prime = self.v(s_prime, second_hidden).squeeze(1)
            td_target = r + self.gamma * v_prime * done_mask
            v_s = self.v(s, first_hidden).squeeze(1)
            delta = td_target - v_s
            delta = delta.detach().numpy()
            
            advantage_lst = []
            advantage = 0.0
            
            for item in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + item[0]
                advantage_lst.append([advantage])
            
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi, _ = self.pi(s, first_hidden)
            pi_a = pi.squeeze(1).gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == log(exp(a)-exp(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v_s, td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)
            self.optimizer.step()
            
    
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
import random
import joblib
from sklearn.ensemble import RandomForestRegressor
import os
from evaluate import evaluate_HIV

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.
# DQN
state_dim = env.observation_space.shape[0]
n_action = env.action_space.n 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, nb_neurons, n_action):
        super(DuelingDQN, self).__init__()
        self.base = nn.Sequential(nn.Linear(state_dim, nb_neurons),
                                  nn.SiLU(),
                                  nn.Linear(nb_neurons, nb_neurons),
                                  nn.SiLU(),
                                  nn.Linear(nb_neurons, nb_neurons),
                                  nn.SiLU())
        self.value = nn.Sequential(nn.Linear(nb_neurons, nb_neurons),
                                   nn.SiLU(),
                                   nn.Linear(nb_neurons, nb_neurons),
                                   nn.SiLU(),
                                   nn.Linear(nb_neurons, 1))
        self.advantage = nn.Sequential(nn.Linear(nb_neurons, nb_neurons),
                                       nn.SiLU(),
                                       nn.Linear(nb_neurons, nb_neurons),
                                       nn.SiLU(),
                                       nn.Linear(nb_neurons, n_action))
        
    def forward(self, x):
        y = self.base(x)
        values = self.value(y)
        advantages = self.advantage(y)
        qvals = values + (advantages - advantages.mean(1, keepdim=True))
        return qvals


class ProjectAgent:
    def __init__(self):
        self.save_name = 'saved_DQN_3.pth'
        nb_neurons=512
        self.DQN = DuelingDQN(state_dim, nb_neurons, n_action).to(device)
        
        self.nb_actions = env.action_space.n
        self.max_episode = 300
        self.start_saving = 100
        self.gamma = 0.98
        self.batch_size = 800
        buffer_size = 100000
        #self.memory = ReplayBuffer(buffer_size,device)
        self.epsilon_max = 1.
        self.epsilon_min = 0.01
        self.epsilon_stop = 98*200
        self.epsilon_delay = 2*200
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop 
        self.target_model = deepcopy(self.DQN).to(device)
        self.criterion = torch.nn.SmoothL1Loss()
        lr = 0.001
        self.optimizer = torch.optim.Adam(self.DQN.parameters(), lr=lr) # RMSProp ?
        self.nb_gradient_steps = 3
        self.update_target_strategy = 'replace'
        self.update_target_freq = 400
        self.update_target_tau = 0.005
        
        #self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.epsilon_delay+self.epsilon_stop,
        #                                                    num_training_steps=self.max_episode*200, lr_max=1., lr_min=0.5)
        

        self.episode_returns = []
        #self.temperature = 0.03
        #self.alpha = 0.9
        #self.l_0 = -1

        #self.best_DQN_state_dict = self.DQN.state_dict()

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D, weights, idxes = self.memory.sample(self.batch_size, 0.4)
            selected_action = self.DQN(X).max(1)[1].detach()
            QYmax = self.target_model(Y).detach().gather(1, selected_action.unsqueeze(1)).squeeze()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)   # Base
            """q_thetabarre = self.target_model(Y).detach()
            pi_thetabarre = F.softmax(q_thetabarre/self.temperature, dim=1)
            s = torch.sum(pi_thetabarre * (q_thetabarre - (self.temperature*F.log_softmax(q_thetabarre/self.temperature, dim=1))).clip(self.l_0, 0), dim=1)
            ln_pi_thetabarre_atst = F.log_softmax(self.target_model(X).detach()/self.temperature, dim=1)[torch.arange(self.batch_size), A.detach().cpu().long()]
            update = torch.addcmul(R + self.alpha*(self.temperature*ln_pi_thetabarre_atst).clip(self.l_0, 0), 
                                   1-D, 
                                   s, 
                                   value=self.gamma)   # Munchausen"""
            QXA = self.DQN(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = (torch.tensor(weights, device=device) * self.criterion(QXA, update.unsqueeze(1))).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
            priorities = ((QXA - update.unsqueeze(1)).abs() + 1e-6).detach().cpu().numpy().flatten()
            self.memory.update_priorities(idxes, priorities)
    
    def train(self, env):
        self.episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        best_validation_score = 0
        while episode < self.max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.greedy_action(self.DQN, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            if len(self.memory) > self.batch_size:
                self.lr_scheduler.step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.DQN.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.DQN.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                if episode > self.start_saving:
                    validation_score = evaluate_HIV(agent=self, nb_episode=1)
                    print("Episode ", '{:3d}'.format(episode), 
                        ", epsilon ", '{:6.2f}'.format(epsilon), 
                        ", batch size ", '{:5d}'.format(len(self.memory)), 
                        ", episode return ", '{:e}'.format(episode_cum_reward),
                        ", validation score ", '{:e}'.format(validation_score),
                        sep='')
                    
                    if validation_score>best_validation_score:
                        best_validation_score = validation_score
                        self.save('')
                        #self.best_DQN_state_dict = self.DQN.state_dict()
                        print('Replaced best DQN')
                else:
                    print("Episode ", '{:3d}'.format(episode), 
                        ", epsilon ", '{:6.2f}'.format(epsilon), 
                        ", batch size ", '{:5d}'.format(len(self.memory)), 
                        ", episode return ", '{:e}'.format(episode_cum_reward),
                        sep='')

                state, _ = env.reset()
                self.episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state
        return self.episode_return
    
    def greedy_action(self, network, state):
        device = "cuda" if next(network.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = network(torch.Tensor(state).unsqueeze(0).to(device))
            return torch.argmax(Q).item()
        
    def act(self, observation, use_random=False):
        a = self.greedy_action(self.DQN, observation)
        return a

    def save(self, path):
        torch.save(self.DQN.state_dict(), path + self.save_name)

    def load(self):
        self.DQN.load_state_dict(torch.load(os.path.join(os.getcwd(), 'src/' + self.save_name), map_location=torch.device('cpu')))
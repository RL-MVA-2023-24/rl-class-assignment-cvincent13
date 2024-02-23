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

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.
state_dim = env.observation_space.shape[0]
n_action = env.action_space.n 
nb_neurons=24

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQN config
config = {'nb_actions': env.action_space.n,
          'learning_rate': 0.001,
          'gamma': 0.95,
          'buffer_size': 1000000,
          'epsilon_min': 0.01,
          'epsilon_max': 1.,
          'epsilon_decay_period': 1000,
          'epsilon_delay_decay': 20,
          'batch_size': 20}


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self):
        self.Q = None

    def act(self, observation, use_random=False):
        a = self.greedy_action(observation)
        return a

    def save(self, path):
        joblib.dump(self.Q, path + 'saved_Q.joblib')

    def load(self):
        self.Q = joblib.load(os.path.join(os.getcwd(), 'src/saved_Q.joblib'))

    def greedy_action(self,s):
        Qsa = []
        for a in range(n_action):
            sa = np.append(s,a).reshape(1, -1)
            Qsa.append(self.Q.predict(sa))
        return np.argmax(Qsa)
    
    def train(self, horizon, nb_iter, gamma):
        S,A,R,S2,D = self.collect_samples(env, horizon)
        Qfunctions = self.rf_fqi(S, A, R, S2, D, nb_iter, n_action, gamma)
        self.Q = Qfunctions[-1]

    def collect_samples(self, env, horizon, disable_tqdm=False, print_done_states=False):
        s, _ = env.reset()
        #dataset = []
        S = []
        A = []
        R = []
        S2 = []
        D = []
        for _ in range(horizon):
            a = env.action_space.sample()
            s2, r, done, trunc, _ = env.step(a)
            #dataset.append((s,a,r,s2,done,trunc))
            S.append(s)
            A.append(a)
            R.append(r)
            S2.append(s2)
            D.append(done)
            if done or trunc:
                s, _ = env.reset()
                if done and print_done_states:
                    print("done!")
            else:
                s = s2
        S = np.array(S)
        A = np.array(A).reshape((-1,1))
        R = np.array(R)
        S2= np.array(S2)
        D = np.array(D)
        return S, A, R, S2, D

    def rf_fqi(self, S, A, R, S2, D, iterations, nb_actions, gamma, disable_tqdm=False):
        nb_samples = S.shape[0]
        Qfunctions = []
        SA = np.append(S,A,axis=1)
        for iter in range(iterations):
            if iter==0:
                value=R.copy()
            else:
                Q2 = np.zeros((nb_samples,nb_actions))
                for a2 in range(nb_actions):
                    A2 = a2*np.ones((S.shape[0],1))
                    S2A2 = np.append(S2,A2,axis=1)
                    Q2[:,a2] = Qfunctions[-1].predict(S2A2)
                max_Q2 = np.max(Q2,axis=1)
                value = R + gamma*(1-D)*max_Q2
            Q = RandomForestRegressor()
            Q.fit(SA,value)
            Qfunctions.append(Q)
        return Qfunctions

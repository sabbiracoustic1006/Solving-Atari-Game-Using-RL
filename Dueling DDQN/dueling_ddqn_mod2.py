# -*- coding: utf-8 -*-
"""
Created on Thu May 26 11:32:55 2022

@author: user
"""

import numpy as np
import torch as T
from network import DuelingDeepQNetworkMod
from buffer import ReplayBufferMod
from torch.nn import functional as F

class DuelingDDQNAgentMod(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, alpha=0.5,  algo=None, env_name=None, chkpt_dir='saved_models/dqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.alpha = alpha
        self.memory = ReplayBufferMod(mem_size, input_dims, n_actions)

        self.q_eval = DuelingDeepQNetworkMod(self.lr, self.n_actions,
                        input_dims=self.input_dims,
                        name=self.env_name+'_'+self.algo+'_q_eval',
                        chkpt_dir=self.chkpt_dir)
        self.q_next = DuelingDeepQNetworkMod(self.lr, self.n_actions,
                        input_dims=self.input_dims,
                        name=self.env_name+'_'+self.algo+'_q_next',
                        chkpt_dir=self.chkpt_dir)

    def store_transition(self, state, action, reward, state_, done, n_pellets, n_ppil, ghost_state):
        self.memory.store_transition(state, action, reward, state_, done, n_pellets, n_ppil, ghost_state)

    def sample_memory(self):
        state, action, reward, new_state, done, env_states = \
                                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)
        env_states = T.tensor(env_states).to(self.q_eval.device)

        return states, actions, rewards, states_, dones, env_states

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = np.array([observation], copy=False, dtype=np.float32)
            state_tensor = T.tensor(state).to(self.q_eval.device)
            _, advantages = self.q_eval.forward(state_tensor)[:2]

            action = T.argmax(advantages).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def replace_target_network(self):
        if self.replace_target_cnt is not None and \
           self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min
                               
    def auxillary_task_loss(self, n_pellet, n_ppil, ghost_state, env_states):
        # loss = F.cross_entropy(n_pellet, env_states[:,0]) + \
        idx = env_states[:,2] != -1      
        loss = F.cross_entropy(n_ppil[idx], env_states[idx,1]) + \
               F.cross_entropy(ghost_state[idx], env_states[idx,2])
    
        return loss
    
    @T.no_grad()
    def calculate_accuracy(self, n_pellet, n_ppil, ghost_state, env_states):
        batch_size = len(n_ppil)
        # acc_n_pellet = sum(env_states[:,0] == n_pellet.argmax(1)).float()/ batch_size
        acc_n_ppil = sum(env_states[:,1] == n_ppil.argmax(1)).float()/ batch_size
        acc_ghost_state = sum(env_states[:,2] == ghost_state.argmax(1)).float()/ batch_size
        return 0, acc_n_ppil, acc_ghost_state

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones, env_states = self.sample_memory()
        indices = np.arange(self.batch_size)

        V_s, A_s = self.q_eval.forward(states)[:2]
        V_s_, A_s_ = self.q_next.forward(states_)[:2]

        V_s_eval, A_s_eval, n_ppil, ghost_state = self.q_eval.forward(states_)

        q_pred = T.add(V_s,
                        (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]

        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))

        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1,keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)
        q_next[dones] = 0.0

        q_target = rewards + self.gamma*q_next[indices, max_actions]

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss = loss + self.alpha * self.auxillary_task_loss(None, n_ppil, ghost_state,
                                                            env_states)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
        
        accs = self.calculate_accuracy(None, n_ppil, ghost_state,
                                        env_states)
        
        # self.alpha = max(0.1, self.alpha - 1e-4)
        return accs

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()
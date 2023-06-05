# -*- coding: utf-8 -*-
"""
Created on Tue May 24 22:56:11 2022

@author: user
"""

import numpy as np
import torch as T
from network import DeepQNetworkMod
from buffer import ReplayBufferMod
from torch.nn import functional as F

class DDQNAgentMod(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, alpha=0.5, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, algo=None, env_name=None, chkpt_dir='saved_models/ddqn'):
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

        self.q_eval = DeepQNetworkMod(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name=self.env_name+'_'+self.algo+'_q_eval',
                                    chkpt_dir=self.chkpt_dir)
        self.q_next = DeepQNetworkMod(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name=self.env_name+'_'+self.algo+'_q_next',
                                    chkpt_dir=self.chkpt_dir)

    def store_transition(self, state, action, reward, state_, done, n_pellets, n_ppil, ghost_state):
        self.memory.store_transition(state, action, reward, state_, done, n_pellets, n_ppil, ghost_state)

    def sample_memory(self):
        state, action, reward, new_state, done, env_states = \
                                self.memory.sample_buffer(self.batch_size)

        states = T.from_numpy(state).to(self.q_eval.device)
        rewards = T.from_numpy(reward).to(self.q_eval.device)
        dones = T.from_numpy(done).to(self.q_eval.device)
        actions = T.from_numpy(action).to(self.q_eval.device)
        states_ = T.from_numpy(new_state).to(self.q_eval.device)
        env_states = T.from_numpy(env_states).to(self.q_eval.device)

        return states, actions, rewards, states_, dones, env_states

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.from_numpy(observation).float().to(self.q_eval.device).unsqueeze(0)
            actions, n_ppil, gs = self.q_eval.forward(state)
            action = T.argmax(actions).item()
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
        loss = F.cross_entropy(n_ppil, env_states[:,1], label_smoothing=0.2) + \
               F.cross_entropy(ghost_state, env_states[:,2], label_smoothing=0.2)
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
        

        q_pred = self.q_eval.forward(states)[0][indices, actions]
        q_next = self.q_next.forward(states_)[0]
        q_eval, n_ppil, ghost_state = self.q_eval.forward(states_)
        # return pred_env_states
        # print(pred_env_states)
        max_actions = T.argmax(q_eval, dim=1)
        # print(dones, q_next[dones].shape)
        done_mul = T.ones_like(q_next, device=self.q_eval.device)
        done_mul[dones] = 0.0
        
        q_next = q_next*done_mul

        q_target = rewards + self.gamma*q_next[indices, max_actions]
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        # print('without', loss)
        loss = loss + self.alpha * self.auxillary_task_loss(None, n_ppil, ghost_state,
                                                            env_states)
        # print('with', loss)
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
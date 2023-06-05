# -*- coding: utf-8 -*-
"""
Created on Fri May 20 08:16:12 2022

@author: user
"""

import collections, random, torch
import cv2, os
import numpy as np
import matplotlib.pyplot as plt
import gym

def SEED_EVERYTHING(seed_val=0):
  random.seed(seed_val)
  np.random.seed(seed_val)
  torch.manual_seed(seed_val)
  torch.cuda.manual_seed_all(seed_val)

def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)
    
def plot_compare(scores_arr, label_arr, window=50):
    plt.figure()
    running_avgs = [np.zeros(len(scores_arr[i])) for i in range(len(scores_arr))] 
    for i in range(len(running_avgs[0])):
        for score_idx, running_avg in enumerate(running_avgs):
            running_avg[i] = np.mean(scores_arr[score_idx][max(0, i-window):(i+1)])
    
    for i in range(len(scores_arr)):    
        plt.plot(np.arange(len(running_avgs[i])), running_avgs[i], label=label_arr[i])
        
    plt.title(f'Running average of previous {window} scores')
    plt.legend()
    
    if not os.path.exists('plots'):
        os.makedirs('plots', exist_ok=True)
        
    plt.savefig(f'plots/{label_arr[0]} vs {label_arr[1]}.PNG')
    
def paths_to_plot_compare(paths, window=50, samples=1000, labels=None):
    scores = [np.load(f'scores/{path}')[:samples] for path in paths]
    if labels is None:
        labels = [path.split(os.sep)[-1].replace('.npy','') for path in paths]
    
    plot_compare(scores, labels, window=window)
    
def plot_curve(scores, figure_file, window=50):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-window):(i+1)])
    plt.plot(np.arange(len(running_avg)), running_avg)
    plt.title(f'Running average of previous {window} scores')
    
    if not os.path.exists('plots'):
        os.makedirs('plots', exist_ok=True)
    
    plt.savefig(figure_file)
    
# def plot_curve(scores, fname):
#     plt.plot(np.arange(len(scores)))

class RepeatActionAndMaxFrame(gym.Wrapper):
    """ modified from:
        https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter06/lib/wrappers.py
    """
    def __init__(self, env=None, repeat=4):
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.frame_buffer = np.zeros_like((2,self.shape))

    def step(self, action):
        t_reward = 0.0
        done = False
        for i in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            t_reward += reward
            idx = i % 2
            self.frame_buffer[idx] = obs
            if done:
                break

        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return max_frame, t_reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.frame_buffer = np.zeros_like((2,self.shape))
        self.frame_buffer[0] = obs
        return obs

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, shape, env=None, crop_frame=False, color=False):
        super(PreprocessFrame, self).__init__(env)
        self.shape=(shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low=0, high=1.0,
                                              shape=self.shape,dtype=np.float32)
        self.crop_frame = crop_frame
        self.color = color
        
    def observation(self, obs):
        if not self.color:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

        if self.crop_frame:
            obs = obs[:-40]
        
        new_obs = cv2.resize(obs, self.shape[1:],
                                    interpolation=cv2.INTER_AREA)
        
        if not self.color:
            new_obs = np.expand_dims(new_obs, -1)
        
        # print(new_obs.shape)
        # print(new_obs.shape)
        # print(resized_screen.shape)

        # new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        # print('after resize', new_obs.shape)
        # new_obs = np.swapaxes(new_obs, 2, 0)
        
        new_obs = np.transpose(new_obs, (2,0,1))
        # print('after swap',new_obs.shape)
        new_obs = new_obs / 255.0
        # print(new_obs)
        # print('hap')
        return new_obs

class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(
                             env.observation_space.low.repeat(n_steps, axis=0),
                             env.observation_space.high.repeat(n_steps, axis=0),
                             dtype=np.float32)
        self.stack = collections.deque(maxlen=n_steps)

    def reset(self):
        self.stack.clear()
        observation = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)
        
        
        # print(np.concatenate(self.stack, axis=0).shape)
        # print(self.observation_space.low.shape)
        return np.concatenate(self.stack, axis=0)

    def observation(self, observation):
        self.stack.append(observation)
        obs = np.concatenate(self.stack, axis=0)
        # print(obs.shape)

        return obs

def make_env(env_name, name_exp, crop_frame=False, color=False, shape=(84,84,1), skip=4):
    env = gym.make(env_name)
    env = gym.wrappers.RecordVideo(env, f"videos/{name_exp}")
    env = RepeatActionAndMaxFrame(env, skip)
    env = PreprocessFrame(shape, env, crop_frame, color)
    env = StackFrames(env, skip)

    return env


def get_modified_reward(reward, frame_cntr, life_lost=False):
    if frame_cntr > 22 and reward == 0:
        reward = -1
        
    if life_lost:
        reward = -200
    
    return reward

def get_env_state(reward, n_pellets, n_ppil, ghost_state, ghost_weak_cntr):
    if reward == 10:
        n_pellets -= 1
    
    if reward >= 50 and reward < 100:
        n_ppil -= 1
        ghost_weak_cntr = 0
        
        if reward > 50:
            n_pellets -= (reward - 50) // 10
    
    elif reward >= 100:
        n_pellets -= (reward % 100) // 10
        
    if 0 <= ghost_weak_cntr <= 40:
        ghost_state = 1
        # ghost_state = 1 if 3 <= ghost_weak_cntr <= 35 else -1
        # if ghost_weak_cntr == 3:
        #     n_ppil -= 1
    else:
        ghost_state = 0
        ghost_weak_cntr = -1
        
    return n_pellets, n_ppil, ghost_state, ghost_weak_cntr
     
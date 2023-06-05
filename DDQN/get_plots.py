# -*- coding: utf-8 -*-
"""
Created on Mon May 30 21:19:58 2022

@author: user
"""

from utils import paths_to_plot_compare

samples = 5000
window = 200

labels = ['DDQN', 'DDQN with reward modification']
# plot for DDQN
paths_to_plot_compare(['DDQN_baseline.npy', 'DDQN_Reward_Mod.npy'], labels=labels,
                      samples=samples, window=window)


#%%
f3 = 'DDQN_baseline.npy'
f4 = 'DDQN_Mod2_alpha_0_5.npy'

labels = ['DDQN', 'DDQN with ghost state estimation']

paths_to_plot_compare([f3, f4], samples=samples,
                      window=window, labels=labels)



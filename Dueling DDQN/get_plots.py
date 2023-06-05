# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 02:41:18 2022

@author: user
"""


from utils import paths_to_plot_compare

f1 = 'DuelDDQN_baseline.npy'
f2 = 'DuelDDQN_Reward_Mod.npy'
samples = 5000
window = 200
labels = ['DuelDDQN',
          'DuelDDQN with reward modification']

paths_to_plot_compare([f1, f2], samples=samples,
                      window=window, labels=labels)


f3 = 'DuelDDQN_baseline.npy'
f4 = 'DuelDDQN_Mod2_alpha_0_25.npy'

labels = ['DuelDDQN', 'DuelDDQN with ghost state estimation']

paths_to_plot_compare([f3, f4], samples=samples,
                      window=window, labels=labels)
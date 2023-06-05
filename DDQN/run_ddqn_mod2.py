# -*- coding: utf-8 -*-
"""
Created on Tue May 24 23:12:05 2022

@author: user
"""

import os
import argparse
import numpy as np
from ddqn_mod2 import DDQNAgentMod
from utils import plot_curve, make_env, get_modified_reward, get_env_state, SEED_EVERYTHING

def train(args):
    name_exp = args.name_exp
    crop_frame = args.crop_frame
    color = args.color
    reward_mod = args.reward_mod
    
    os.makedirs('scores', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    
    SEED_EVERYTHING()
    
    if color:
        env = make_env('MsPacman-v0', name_exp, crop_frame, color, shape=(84,84,3))
    else:
        env = make_env('MsPacman-v0', name_exp, crop_frame, color, shape=(84,84,1))
        
    
    env.seed(0)
    best_score = -np.inf
    load_checkpoint = args.eval
    n_games = 100 if args.eval else args.n_games 
    epsilon = 0.05 if args.eval else 1.0
    agent = DDQNAgentMod(gamma=args.gamma, epsilon=epsilon, lr=args.lr,
                         input_dims=(env.observation_space.shape),
                         n_actions=env.action_space.n, alpha=args.alpha, mem_size=50000, eps_min=0.1,
                         batch_size=args.batch_size, replace=args.tau, eps_dec=args.eps_dec,
                         chkpt_dir=f'saved_models/{name_exp}', algo='DDQNAgent',
                         env_name='MsPacman-v0')

    if load_checkpoint:
        agent.load_models()

    n_steps = 0
    scores, eps_history, steps_array = [], [], []
    

    
    for i in range(n_games):
        done = False
        observation = env.reset()
        

        score = 0
        num_life = 3
        n_pellets = 150
        n_ppil = 4
        ghost_state = 0
        os.makedirs(f'visualize/episode_{i}', exist_ok=True)
        ghost_weak_cntr = -1
        frame_cntr = 0
        while not done:            
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            
            
            n_pellets, n_ppil, ghost_state, ghost_weak_cntr = get_env_state(reward, n_pellets, 
                                                                            n_ppil, ghost_state, 
                                                                            ghost_weak_cntr)
            
            
 
            if ghost_weak_cntr != -1:
                ghost_weak_cntr += 1
      
                
            if reward_mod:
                reward = get_modified_reward(reward, frame_cntr)
            
                if info['lives'] < num_life:
                    num_life = info['lives']
                    reward = get_modified_reward(reward, frame_cntr, True)
            
            frame_cntr += 1
                

                
            if not load_checkpoint:
                agent.store_transition(observation, action,
                                     reward, observation_, int(done), 
                                     n_pellets, n_ppil, ghost_state)
                accs = agent.learn()
            else:
                accs = [0,0,0]
                
            observation = observation_
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)
        
        

        avg_score = np.mean(scores[-50:])
        print('episode: ', i,'score: ', score,
             ' average score %.1f' % avg_score, 'best score %.2f' % best_score, f'acc_n_pell:{accs[0]:.3f}, acc_n_ppil:{accs[1]:.3f}, acc_gs:{accs[2]:.3f}',
             'epsilon %.2f' % agent.epsilon, 'steps', n_steps)


        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)

        
    if not load_checkpoint:
        plot_curve(scores, f'plots/{name_exp}')
        np.save(f'scores/{name_exp}.npy', scores)
    else:
        print('Avg of 100 scores:',np.mean(scores))

        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name_exp', default='DDQN_Mod2_alpha_0_5', type=str, help='name of the experiment')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size to be used for training')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate to use for training')
    parser.add_argument('--eps_dec', default=5e-6, type=float, help='epsilon decay rate to use for training')
    parser.add_argument('--gamma', default=0.99, type=float, help='discount factor')
    parser.add_argument('--alpha', default=0.5, type=float, help='loss weighting to use for ghost state loss')
    parser.add_argument('--tau', default=10000, type=int, help='perform copy after tau steps')
    parser.add_argument('--n_games', default=5000, type=int, help='number of episodes for training the agent')
    parser.add_argument('--crop_frame', action='store_true', help='whether to crop frame')
    parser.add_argument('--color', action='store_true', help='whether to use RGB images for training')
    parser.add_argument('--eval', action='store_true', help='whether to evaluate a trained model')
    parser.add_argument('--reward_mod', action='store_true', help='whether to perform reward modification (modification 1)')
    args = parser.parse_args()
    
    print(args)
    
    train(args)
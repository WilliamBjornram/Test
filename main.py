from classEnv import GameEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.wrappers import ActionMasker
from graphClass import Graph
import os
import numpy as np
import gymnasium
import random

def main(graph):
    
    env = GameEnv(graph)
    env = ActionMasker(env, mask_fn)
 
    
    log_path = os.path.join('Training', 'Logs')
    #model = MaskablePPO("MlpPolicy", env,learning_rate=0.005, n_steps=1024, gamma= 0.9,gae_lambda=0.99, clip_range=0.3, verbose=1)
    model = MaskablePPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model_path = os.path.join("Training", "Saved Models", "PPO_50k")
    model.save(model_path)
    
    #model = MaskablePPO.load(model_path)
    
    """
    temp = 0
    info = 0
    episodes = 10
    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        
        score = 0

        while not done:
            env.render()
            action_mask = env.action_masks()
            valid_actions = [i for i, allowed in enumerate(action_mask) if allowed]
            action = random.choice(valid_actions)
            print(action)
            n_state, reward, done, temp, info = env.step(action)
            score+=reward
        print('Episode:{} Score:{}'.format(episode, score))
    env.close()
    """
    answer = input("Evaluate model? (Ja/Nej): ")
    while answer.strip().lower() == "ja":
        
        if answer.strip().lower() == "ja":
            episoder = input("Hur många ep: ")
            env.render()
            mean, std = evaluate_policy(model, env, n_eval_episodes=int(episoder), warn=False)
            print("Medel: " + str(mean))
            print("avvikelse: " + str(std))
            answer = input("Evaluate model igen? (Ja/Nej): ")
    
    


def mask_fn(env: gymnasium.Env):
    return env.action_mask()

if __name__ == "__main__":
    file = r"C:\Users\William\Desktop\Kex-jobb\01-31-kod\graph2.csv"
    graph = Graph(file)
    main(graph)

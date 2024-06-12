#!/usr/bin/env python
# coding: utf-8

import pandas as pd 
import numpy as np
import seaborn as sns 
import random
import gym
from gym import spaces 
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

file_path = r"C:\Users\4967\OneDrive - Wavestone Germany Group\Dokumente\problem_klein.xlsx"
data = pd.read_excel(file_path)

# Data cleaning, formatting and encoding (categorical variables)

data.replace(["NA", "", "null", "Na", "N/A"], np.nan, inplace=True)

label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))
    label_encoders[column] = le

# Train 75%, Test 25% 

train_data, test_data = train_test_split(data, test_size=0.25, random_state=0) 

print(data.head())


# Environment and Actions for RL
class SampleEnvironment(gym.Env):
    def __init__(self, data):
        super(SampleEnvironment, self).__init__()
        self.data = data
        self.original_data = self.data.copy()
        self.iteration = 0
        self.finish = False
        self.current_col = 0
        self.num_col = len(self.data.columns)
        self.num_row = len(self.data)
        
        self.action_space = spaces.Discrete(self.num_row)

        # observation space: flattened data for consistent state representation (many RL algorithms expect fixed size input)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_row, self.num_col), dtype=np.float32) 
    
    def reset(self):
        self.data = self.original_data.copy()
        self.iteration = 0 
        self.current_col = 0
        return self.data.values
        
    def step(self, action): 
        row_i = action 
        column_i = self.current_col
        if pd.isna(self.data.iloc[row_i, column_i]):
            self.data.iloc[row_i, column_i] = self.impute_value(row_i, column_i)
        self.iteration += 1

        # move to next column 
        if self.iteration >= self.num_row:
            self.iteration = 0 
            self.current_col += 1
        
        finished = self.current_col >= self.num_col 
        reward = self.calculate_reward()
        if finished: 
            self.finish = True 
        return self.data.values, reward, finished, {}

    def impute_value(self, row_i, column_i):
        # Iterative Imputer with Random Forest Regressor as base model 
        target_col = self.data.columns[column_i]
        with_values = self.data.drop(columns=[target_col])

        imputer = IterativeImputer(estimator=RandomForestRegressor(), max_iter=10, random_state=0)
        imputed_data = imputer.fit_transform(self.data)
        return imputed_data[row_i, column_i]
                
    def calculate_reward(self):
        # Simple reward: negative count of remaining NaNs
        return -self.data.isna().sum().sum()

class Agent: 
    def __init__(self, n_action, n_state): 
        self.epsilon_start = 0.9 # exploration 
        self.epsilon_end = 0.05 # min exploration 
        self.epsilon_decay = 0.995 # exploration decay
        self.discount = 0.95 
        self.lr = 1e-4 # learning rate
        self.gamma = 0.99 # exploitation 
        self.q_table = np.zeros((n_state, n_action))
        self.n_action = n_action 
        self.n_state = n_state

    def take_action(self, state):
        if random.randrange(0, 1, 0.00001) < self.epsilon_start():
            return np.random.choice(self.n_action)
        return np.argmax(self.q_table[state, :])

    def learn(self, state, next_state, action, reward, done):
        best_next_action = np.argmax(self.q_table[state, :])
        
            
    def step(self, env: SampleEnvironment):
        action = random.choice(range(env.num_row))
        _, reward, finished, _ = env.step(action)
        self.total_rewards += reward
        return finished

if __name__ == "__main__": 
    env = SampleEnvironment(train_data)
    agent = Agent()

    env.reset()
    i = 0

    while not env.finish and i < 1000:  # limit iterations to prevent infinite loop
        i += 1
        finished = agent.step(env)
        if finished:
            break
    print("Total Rewards:", agent.total_rewards)

import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import random
import math as m
import gym
from gym import spaces 
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

file_path = r"C:\Users\4967\OneDrive - Wavestone Germany Group\Dokumente\problem_klein.xlsx"
solved_file_path = r"C:\Users\4967\OneDrive - Wavestone Germany Group\Dokumente\solved_klein.xlsx"
data = pd.read_excel(file_path, header=0)

solved_data = pd.read_excel(solved_file_path)

# Data formatting  
data.replace(["NA", "", "null", "Na", "N/A", "na"], np.nan, inplace=True)
print(data)

# VALIDITY: invalid values are stored in dictionary - DATA TYPES 
invalid_indices_dict = {}
for column in data.columns:
    original_nan_mask = data[column].isna()
    if pd.api.types.is_numeric_dtype(data[column]):
        coerced_column = pd.to_numeric(data[column], errors='coerce')  # convert numeric columns to float, coerce non-numeric to NaN
        coerced_nan_mask = coerced_column.isna() # create mask for original nan values and updated after errors = coerce 
        coerced_values_mask = coerced_nan_mask & ~original_nan_mask # extract coerced nan values 
        invalid_indices = data[coerced_values_mask].index 
        for idx in invalid_indices: # append index of coerced values to dictionary 
            if idx not in invalid_indices_dict:
                invalid_indices_dict[idx] = []
            invalid_indices_dict[idx].append({
                'column': column,
                'previous_value': data.at[idx, column]
            })
        data[column] = coerced_column # replace original column with coerced column 

        #if 'age' in column.lower():
        #    data[column] = np.where((data[column] >= 0) & (data[column] <= 100), data[column], np.nan)
    else:
        for row_idx, value in data[column].items():
            try:
                float_value = float(value) # attempt to convert to float
                if not m.isnan(float_value):
                    if row_idx not in invalid_indices_dict: # store row and column index, as well as previous value, in dictionary if it succeeds
                        invalid_indices_dict[row_idx] = []
                    invalid_indices_dict[row_idx].append({
                        'column': column,
                        'previous_value': value,
                    })
                    data.at[row_idx, column] = np.nan  # convert numeric value in non-numeric column to NaN
            except ValueError:
                pass
print(data) 

# Encoding of categorical variables as numeric: decode after ML imputation to get original dataset 
ordinal_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    series = data[column]
    indices_to_encode = series.notna()
    series_filled = series.fillna('missing')
    ordinal_encoder = OrdinalEncoder()
    encoded_values = ordinal_encoder.fit_transform(series_filled[indices_to_encode].values.reshape(-1, 1)).flatten()
    series_copy = series.copy()
    series_copy.loc[indices_to_encode] = encoded_values
    data[column] = series_copy
    ordinal_encoders[column] = ordinal_encoder

# Encoding solved dataset
for column in solved_data.select_dtypes(include=['object']).columns:
    if column in ordinal_encoders:
        ordinal_encoder = ordinal_encoders[column]
        series = solved_data[column]
        encoded_values = ordinal_encoder.transform(series.values.reshape(-1, 1)).flatten()
        solved_data[column] = encoded_values
    
# Train 75%, Test 25%: after training, use unseen data in test subset to estimate algorithm accuracy and validity 
train_data, test_data = train_test_split(data, test_size=0.25, random_state=0, shuffle=False)
print(data)
print(invalid_indices_dict)

# Data visualisation before data cleaning process


# Environment and Actions for RL
class Environment(gym.Env): # OpenAI Gym Environment Inheritance 
    def __init__(self, data, solved_data):
        super(Environment, self).__init__()
        self.data = data
        self.solved_data = solved_data
        self.original_data = self.data.copy()
        self.iteration = 0
        self.finish = False
        self.current_col = 0
        self.num_col = len(self.data.columns) 
        self.num_row = len(self.data) 
        self.inaccurate_numbers = []
        self.previous_na = self.data.isna().sum().sum()

        # Action space describes the possible values that actions can take (finite) 
        self.action_space = spaces.Discrete(3)

        # Observation space (Box) describes valid values that observations can take for consistent state representation  
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_row, self.num_col), dtype=np.float32) 

        # Standardizing data for One-Class SVM outlier detection (accuracy) 
        self.scaler = StandardScaler().fit(self.data.fillna(-9999)) # Standard Scaler cannot handle NaN values; they have to be transformed to out-of-bound number
        self.scaled_data = pd.DataFrame(self.scaler.transform(self.data.fillna(-9999)), columns=self.data.columns) # transform data while maintaining dataframe structure and column names

        self.ocsvm = OneClassSVM(nu=0.01, gamma='scale', kernel='rbf') # nu: upper bound on the fraction of margin errors and support vectors: at most 1% misinterpreted values
        self.ocsvm.fit(self.scaled_data)
        
        # Imputation Techniques 
        self.imputer = IterativeImputer(estimator=RandomForestRegressor(), max_iter=10, random_state=0)
        self.imputer_knn = KNNImputer(n_neighbors=5)
        self.imputer_mean = SimpleImputer(strategy='mean')


    # Reset dataset back to original values after each episode of training 
    def reset(self):
        self.data = self.original_data.copy() 
        self.iteration = 0 
        self.current_col = 0 
        self.previous_na = self.data.isna().sum().sum()
        self.imputed_data = self.imputer.fit_transform(self.data)
        self.imputed_data_knn = self.imputer_knn.fit_transform(self.data)
        self.imputed_data_mean = self.imputer_mean.fit_transform(self.data)
        self.scaled_data = pd.DataFrame(self.scaler.transform(self.imputed_data), columns=self.data.columns)
        return self.data.values.flatten() 

    # Iterate over each row in every column and apply impute_value where NaN, check_accuracy to all and calculate_reward when done with episode
    def step(self, action): 
        row_i = self.iteration 
        column_i = self.current_col
        done = self.current_col == self.num_col - 1 and self.iteration == self.num_row - 1
        if column_i < self.num_col:
            if self.data.iloc[row_i, column_i] == -9999 or pd.isna(self.data.iloc[row_i, column_i]): # Look for NaN values (or that were replaced with -9999)
                try:
                    self.data.iloc[row_i, column_i] = self.impute_value(action, row_i, column_i)    
                except (TypeError, ValueError):
                    pass
                    
            self.check_accuracy() # check if value is accurate (outliers)

        if done: 
            self.finish = True
            reward = self.calculate_reward(action, row_i, column_i)
            return self.data.values.flatten(), reward, done, {}
        
        # Move to next row
        reward = self.calculate_reward(action, row_i, column_i)
        self.iteration += 1
            
        # Move to next column 
        if self.iteration >= self.num_row:
            self.iteration = 0 
            self.current_col += 1
        return self.data.values.flatten(), reward, done, {}
        
    # COMPLETENESS: Impute missing values based on ML imputation method 'Iterative Imputer' using 'Random Forest Regressor', 'K Nearest Neighbour' or the Mean
    def impute_value(self, action, row_i, column_i):        
        self.data.replace(-9999, np.nan, inplace=True)
        if action == 0:
            return self.imputed_data_rf[row_i, column_i]
        elif action == 1:
            return self.imputed_data_knn[row_i, column_i]
        elif action == 2:
            return self.imputed_data_mean[row_i, column_i]
        
    # ACCURACY: To replace inaccurate values (that are disproportionate/ outliers compared to the rest of the data) with predicted values
    def check_accuracy(self): 
        self.scaled_data = pd.DataFrame(self.scaler.transform(self.imputed_data), columns=self.data.columns) # maintain column names

        # One-Class SVM for outlier detection
        self.ocsvm.set_params(nu=0.01)
        outliers_ocsvm = self.ocsvm.fit_predict(self.scaled_data)
        outliers = np.where(outliers_ocsvm == -1)[0] # finds outliers
        self.data[:] = self.scaler.inverse_transform(self.scaled_data) # transform scaled data back to original scale
        
        for index in outliers:
            row_index = index // self.data.shape[1] # calculate row index 
            col_index = index % self.data.shape[1]

            if col_index == self.current_col: 
                for entry in self.inaccurate_numbers:
                    if entry['row'] == row_index and entry['col'] == col_index:
                        self.inaccurate_numbers = [
                            entry for entry in self.inaccurate_numbers
                            if not (entry['row'] == row_index and entry['col'] == col_index)
                        ]
            
                self.inaccurate_numbers.append({ 
                    'row': row_index,
                    'col': col_index,
                    'previous_value': self.data.iloc[row_index, col_index]
                })
                
        # -- TO DO (eventually) -- within given boundary (very obvious outliers), values are either changed through best fit prediction or appended to inaccurate_numbers list
        # IDEA: increase nu parameter linear to exploration factor epsilon (nu increases proportional to exploration rate, so that the agent can learn the optimal nu parameter) 
        
    # Calculate reward based on reduction of NaN, consistency with allowed data types, accuracy of data imputs (-- TO DO --)
    def calculate_reward(self, action, row_i, column_i):
        # Completeness: the less NaN values remaining and the closer they are to the solved dataset, the higher the reward
        current_value = self.data.iloc[row_i, column_i]
        solved_value = self.solved_data.iloc[row_i, column_i]
        
        if pd.isnull(current_value) or current_value == -9999:  # Handle NaN or imputed values
            return 0  # No reward for missing values or already imputed values

        print(f"Imputed Value: {current_value} and Solved Value: {solved_value}")
        diff = abs(current_value - solved_value)
        # Reward based on the closeness of the imputed value to the solved value
        if diff == 0:
            reward = 100  # Maximum reward for exact match
        else:
            reward = 100 / (diff + 1)  # Reward decreases as the difference increases
    
        return reward


class Agent: 
    def __init__(self, n_state, n_action): 
        # values (especially epsilon and gamma) can be adjusted for best outcome (trial and error) 
        self.epsilon = 0.8 # exploration 
        self.min_epsilon = 0.01 # min exploration as exploitation becomes more important
        self.lr = 0.2 # learning rate: adjust Q values to converge towards optimal strategy 
        self.gamma = 0.8 # discounting rate (value of future rewards)
        self.n_state = n_state 
        self.n_action = n_action
        self.q_table = {} # initial Q table for learning (initially all zeros as there are no state-action rewards yet) 
        self.state_index_map = {}
        self.next_state_index = 0
            
    def state_to_index(self, state):
        # Convert state to tuple and map to a unique index
        if isinstance(state, (list, tuple, np.ndarray)): # Check if state is iterable: if it is then it is converted to a tuple 
            state_tuple = tuple(state)
        else:
            state_tuple = (state,) # if it is not then it is converted to a tuple with single instance  
        if state_tuple not in self.state_index_map: # if the state tuple is not already in the state index map dictionary it is added
            self.state_index_map[state_tuple] = self.next_state_index
            self.q_table[self.next_state_index] = np.zeros(self.n_action) # the Q table is updated 
            self.next_state_index += 1 # index count is updated 
        return self.state_index_map[state_tuple]
        
    # Q learning Algorithm with epsilon greedy policy iteration 
    def take_action(self, state):
        state_index = self.state_to_index(state) # state indexing
        if np.random.rand() <= self.epsilon: # epsilon greedy strategy for current state
            return np.random.randint(self.n_action) # exploration epsilon % of the time 
        return np.argmax(self.q_table[state_index]) # otherwise exploit accumulated knowledge: choose maximal value in Q table

    def learn(self, state, next_state, action, reward, done): 
        state_index = self.state_to_index(state) # state indexing
        next_state_index = self.state_to_index(next_state) # next state indexing 
        action = int(action) # convert action to integer 

        # Temporal Difference Learning for Policy Evaluation 
        best_next_action = np.argmax(self.q_table[state_index]) # Greedy strategy for future state
        td_prediction = reward + (self.gamma * self.q_table[state_index][best_next_action] * (1 - done)) # TD prediction formula: reward + discounting rate * max Q(S(t+1), A), (1 - done) to ignore terminal rewards that are irrelevant
        td_error = td_prediction - self.q_table[state_index][action] # TD error formula: TD prediction - current Q(S, A) 
        self.q_table[state_index][action] += self.lr * td_error # TD learning formula: Q(S, A) + TD error * learning rate (alpha)
    
        # decrease exploration to encourage exploitation with each episode 
        if done: 
            self.epsilon = max(self.min_epsilon, self.epsilon * self.gamma) # choose max value between epsilon * discounting rate and 0.01 (convergence towards exploitation)

    def update_state(self, state):
        self.current_state = self.state_to_index(state) # state index updated 


if __name__ == "__main__":
    env = Environment(train_data, solved_data)
    agent = Agent(n_state=env.observation_space.shape[0], n_action=env.action_space.n)
    
    episodes = 1
    for episode in range(episodes):
        state = env.reset()
        agent.update_state(state)
        total_reward = 0
        
        while True:
            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            reward = env.calculate_reward(action, env.iteration, env.current_col)
            agent.learn(state, next_state, action, reward, done) # SARS -> Q learning 
            
            state = next_state
            total_reward += reward
            print(f"Episode: {episode + 1}, Row: {env.iteration}, Column: {env.current_col}, Reward: {reward}")

            if done: 
                print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Exploration Rate: {agent.epsilon}")
                print(f"Invalid Numbers: {invalid_indices_dict}, Inaccurate Numbers: {env.inaccurate_numbers}")
                print(env.data)
                break
    
    np.round(env.data)
    decoded_data = env.data.copy().to_numpy()
    for column in ordinal_encoders:
        col_idx = env.data.columns.get_loc(column)
        ordinal_encoder = ordinal_encoders[column]
        decoded_values = ordinal_encoder.inverse_transform(decoded_data[:, col_idx].astype(int).reshape(-1, 1)).flatten()
        decoded_data[:, col_idx] = decoded_values
    decoded_data = pd.DataFrame(decoded_data, columns=env.data.columns)
    print(decoded_data)

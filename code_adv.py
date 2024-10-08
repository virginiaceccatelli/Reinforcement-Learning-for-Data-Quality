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
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

file_path = FILE PATH 
solved_file_path = FILE PATH 
data = pd.read_excel(file_path, header=0)

solved_data = pd.read_excel(solved_file_path, header=0)

# Data formatting  
data = data.drop(columns=["missing_values"])
data.replace(["NA", "", "null", "Na", "N/A", "na"], np.nan, inplace=True)
print(data)

# Comparison of the two datasets 
def compare_datasets(df1, df2):
    # Check if datasets have the same columns (if they are comparable) 
    if df1.columns.tolist() != df2.columns.tolist():
        raise ValueError("The datasets don't have the same columns.")
    # Total number of cells
    total_cells = df1.size
    # Compare values of cells 
    equal_cells = (df1 == df2).sum().sum()
    # Calculate percentage of similarity 
    similarity_ratio = (equal_cells / total_cells) * 100
    return similarity_ratio

# Percentage of similarity between datasets 
similarity_pre = compare_datasets(data, solved_data)
print(f"The datasets are similar to {similarity_pre:.2f}%")

# FIRST VALIDITY CHECK: check after data types - invalid values are stored in dictionary 
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

# SECOND VALIDITY CHECK: check after point of truth (e.g. age of driver above 85) - adjustable to specific dataset 

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
print(solved_data)
print(invalid_indices_dict)


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
        self.nan_positions = np.argwhere(self.data.isna().values)

        # Action space describes the possible values that actions can take (finite: imputation techniques) 
        self.action_space = spaces.Discrete(2)

        # Observation space (Box) describes valid values that observations can take for consistent state representation  
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_row, self.num_col), dtype=np.float32) 

        # Outliers for Accuracy: nu, gamma and kernel can be adjusted for optimal outcome (trial and error) 
        self.ocsvm = OneClassSVM(nu=0.01, gamma=0.1, kernel='rbf') # nu: upper bound on the fraction of margin errors and support vectors: at most 1% misinterpreted values
        
        # Imputation Techniques initialised 
        self.imputer_rf = IterativeImputer(estimator=RandomForestRegressor(), max_iter=10, random_state=0)
        self.imputer_knn = KNNImputer(n_neighbors=5)
        self.imputed_rf_data = self.imputer_rf.fit_transform(self.data.values.astype(float))
        self.imputed_knn_data = self.imputer_knn.fit_transform(self.data.values.astype(float))

    # Reset dataset back to original values after each episode of training 
    def reset(self):
        self.data = self.original_data.copy() 
        self.iteration = 0 
        self.inaccurate_numbers = []
        self.nan_positions = np.argwhere(self.data.isna().values)
        self.imputed_rf_data = self.imputer_rf.fit_transform(self.data.values.astype(float))
        self.imputed_knn_data = self.imputer_knn.fit_transform(self.data.values.astype(float))
        self.finish = False
        return self.data.values.flatten() 

    # Iterate over each row in every column and apply impute_value where NaN, check_accuracy to all and calculate_reward when done with episode
    def step(self, action): 
        done = False 
        
        for row_i, column_i in self.nan_positions:
            self.impute_value(action, row_i, column_i)
            reward = self.calculate_reward(action, row_i, column_i)
        accuracy = self.check_accuracy()
        self.data = self.data.round()
        done = True                
        return self.data.values.flatten(), reward, done, accuracy, {}
        
    # COMPLETENESS: Impute missing values based on ML imputation method 'Iterative Imputer' using 'Random Forest Regressor' or 'K Nearest Neighbour'
    def impute_value(self, action, row_i, column_i): 
        if action == 0:
            imputed_value = self.imputed_rf_data 
        elif action == 1:
            imputed_value = self.imputed_knn_data

        # Update the DataFrame with imputed values
        self.data.iloc[row_i, column_i] = imputed_value[row_i, column_i]
        
    # ACCURACY: To highlight inaccurate values that are disproportionate/ outliers compared to the rest of the data
    def check_accuracy(self):
        flattened_data = self.data.values.flatten().reshape(-1, 1)
        self.ocsvm.fit(flattened_data)
        predictions = self.ocsvm.predict(flattened_data)
    
        # One-Class SVM for outlier detection
        inliers = np.sum(predictions == 1)
        outliers = np.sum(predictions == -1)
        
        # Calculate the accuracy based on the proportion of inliers
        self.accuracy = (inliers / (inliers + outliers)) * 100
        return self.accuracy
                        
    # Calculate reward based on similarity to solved dataset
    def calculate_reward(self, action, row_i, column_i):
        # Completeness: the less NaN values remaining and the closer they are to the solved dataset, the higher the reward
        total_similarity = 0
        total_values = 0

        for row_i, column_i in self.nan_positions:
            current_value = self.data.iloc[row_i, column_i]
            solved_value = self.solved_data.iloc[row_i, column_i]

            if pd.isnull(current_value):
                return -1  # negative reward calculation for missing values

            current_value_rounded = current_value.round().astype(type(current_value))  # round to whole number
            diff = abs(current_value - solved_value)
            similarity = 1 / (diff + 1)  # the reward increases with similarity (inverse of difference)
            total_similarity += similarity
            total_values += 1

        if total_values == 0:
            return 0  # no valid values to compare

        average_similarity = total_similarity / total_values
        return average_similarity * 100  # Scale to percentage


class Agent: 
    def __init__(self, n_state, n_action): 
        # values can be adjusted for best outcome (trial and error) 
        self.epsilon = 0.5 # exploration 
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
        if np.random.rand() < self.epsilon: # epsilon greedy strategy for current state
            return np.random.randint(self.n_action) # exploration epsilon % of the time 
        return np.argmax(self.q_table[state_index]) # otherwise exploit accumulated knowledge: choose maximal value in Q table

    def learn(self, state, next_state, action, reward, done): 
        state_index = self.state_to_index(state)
        next_state_index = self.state_to_index(next_state) 
        action = int(action) # convert action to integer 

        # Temporal Difference Learning for Policy Evaluation 
        best_next_action = np.argmax(self.q_table[state_index]) # Greedy strategy for future state
        td_prediction = reward + (self.gamma * self.q_table[state_index][best_next_action] * (1 - done)) # TD prediction formula: reward + discounting rate * max Q(S(t+1), A), (1 - done) to ignore terminal rewards that are irrelevant
        td_error = td_prediction - self.q_table[state_index][action] # TD error formula: TD prediction - current Q(S, A) 
        self.q_table[state_index][action] += self.lr * td_error # TD learning formula: Q(S, A) + TD error * learning rate (alpha)

        # decrease exploration to encourage exploitation with each episode 
        if done: 
            self.epsilon = max(self.min_epsilon, self.epsilon * self.gamma) 

    def update_state(self, state):
        self.current_state = self.state_to_index(state) # state index updated 



if __name__ == "__main__":
    env = Environment(data, solved_data)
    agent = Agent(n_state=env.observation_space.shape[0], n_action=env.action_space.n)

    episodes = 100
    for episode in range(episodes):
        state = env.reset()
        agent.update_state(state)
        
        while True:
            action = agent.take_action(state)
            next_state, reward, done, accuracy, _ = env.step(action)
            agent.learn(state, next_state, action, reward, done) # SARS -> Q learning 
            state = next_state
            print(f"Episode: {episode + 1}, Reward: {reward}")

            if done:
                print(f"Episode: {episode + 1}, Accuracy: {accuracy:.2f}%")
                episode_rewards.append(total_reward)
                round(env.data)
                break
    
    # Decode training data (problem dataset)     
    decoded_data = env.data.copy()
    for column in decoded_data.columns:
        if column in ordinal_encoders:
            ordinal_encoder = ordinal_encoders[column]
            col_idx = decoded_data.columns.get_loc(column)
            
            # Inverse transform encoded values to original categorical labels
            encoded_values = decoded_data[column].to_numpy().astype(int)  
            decoded_values = ordinal_encoder.inverse_transform(encoded_values.reshape(-1, 1)).flatten()
            
            # Assign decoded_values back to the corresponding column in decoded_data DataFrame
            decoded_data[column] = decoded_values.astype(str)
    
    columns_to_convert = []
    for column in columns_to_convert:
        decoded_data[column] = decoded_data[column].astype(int)
    print(decoded_data)
    
    # Solved dataset reloaded
    solved_data_original = pd.read_excel(solved_file_path, header = 0) # reload dataset from file path because solved_dataset was encoded
    float_columns = solved_data_original.select_dtypes(include=['float']).columns
    for column in float_columns:
        solved_data_original[column] = solved_data_original[column].astype(int)
    print(solved_data_original) 
        
    # Percentage of similarity between datasets 
    similarity = compare_datasets(decoded_data, solved_data_original)
    print(f"The datasets are similar to {similarity:.2f}%")

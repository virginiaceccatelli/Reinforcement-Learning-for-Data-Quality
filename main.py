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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

file_path = r"C:\Users\4967\OneDrive - Wavestone Germany Group\Dokumente\problem_df.xlsm"
data = pd.read_excel(file_path)

# Data formatting (NA to NaN) 
data.replace(["NA", "", "null", "Na", "N/A", "na"], np.nan, inplace=True)
print(data.head())

# Encoding of categorical variables as numeric: decode after ML imputation to get original dataset 
label_encoders = {}

for column in data.select_dtypes(include=['object']).columns:
    if data[column].nunique() <= 20: 
        series = data[column]
        label_encoder = LabelEncoder()
        data[column] = pd.Series(
            label_encoder.fit_transform(series[series.notnull()]),
            index=series[series.notnull()].index
        )
        label_encoders[column] = label_encoder

# Train 75%, Test 25%: after training, use unseen data in test subset to estimate algorithm accuracy and validity 
train_data, test_data = train_test_split(data, test_size=0.25, random_state=0)
print(data.head())


# Environment and Actions for RL
class Environment(gym.Env): # OpenAI Gym Environment Inheritance 
    def __init__(self, data):
        super(Environment, self).__init__()
        self.data = data
        self.original_data = self.data.copy()
        self.iteration = 0
        self.finish = False
        self.current_col = 0
        self.num_col = len(self.data.columns)
        self.num_row = len(self.data)
        self.previous_na = self.data.isna().sum().sum()

        # Action space describes the possible values that actions can take (finite) 
        self.action_space = spaces.Discrete(self.num_row)

        # Observation space (Box) describes valid values that observations can take for consistent state representation  
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_row, self.num_col), dtype=np.float32) 

        # Standardizing data for One-Class SVM outlier detection (accuracy) 
        self.scaler = StandardScaler().fit(self.data.fillna(-9999)) # OCSVM cannot handle NaN values; they have to be transformed to out-of-bound number
        self.scaled_data = pd.DataFrame(self.scaler.transform(self.data.fillna(-9999)), columns=self.data.columns) # transform data while maintaining dataframe structure and column names

        self.ocsvm = OneClassSVM(nu=0.01) # nu: upper bound on the fraction of margin errors and support vectors
        self.ocsvm.fit(self.scaled_data)
        
        # Initializing IterativeImputer once and reuse
        self.imputer = IterativeImputer(estimator=RandomForestRegressor(), max_iter=10, random_state=0)

    # Reset dataset back to original values after each episode of training 
    def reset(self):
        self.data = self.original_data.copy() 
        self.iteration = 0 
        self.current_col = 0 
        self.previous_na = self.data.isna().sum().sum()
        self.imputed_data = self.imputer.fit_transform(self.data)
        self.scaled_data = pd.DataFrame(self.scaler.transform(self.imputed_data), columns=self.data.columns)
        return self.data.values.flatten() 

    # Iterate over each row in every column and apply impute_value where NaN, check_accuracy to all and calculate_reward when done with episode
    def step(self, action): 
        row_i = action 
        column_i = self.current_col
        if self.data.iloc[row_i, column_i] == -9999 or pd.isna(self.data.iloc[row_i, column_i]): # Look for NaN values (or that were replaced with -9999)
            try:
                imputed_dataframe = self.impute_value()
                self.data.iloc[row_i, column_i] = imputed_dataframe.iloc[row_i, column_i]
            except (TypeError, ValueError) as e:
                print(f"Error: {e}")
                pass
                
        # -- TO DO -- check if data entry is valid, if not apply function 
        self.check_accuracy()
        self.iteration += 1
        
        # Move to next column 
        if self.iteration >= self.num_row:
            self.iteration = 0 
            self.current_col += 1
        
        done = self.current_col >= self.num_col 
        if done: 
            self.finish = True 
            
        reward = self.calculate_reward()
        
        return self.data.values.flatten(), reward, done, {}
        
    # COMPLETENESS: Impute missing values based on ML imputation method 'Iterative Imputer' using 'Random Forest Regressor' 
    def impute_value(self):        
        self.data.replace(-9999, np.nan, inplace=True)
        imputed_data = np.round(self.imputer.fit_transform(self.data))
        imputed_dataframe = pd.DataFrame(imputed_data, columns=self.data.columns)

        # -- TO DO -- round imputed value to closest whole number 
        return imputed_dataframe
    
    # -- TO DO -- function for validity; replacement of impossible inputs (data types) 

    # ACCURACY: To replace inaccurate values (that are disproportionate/ outliers compared to the rest of the data) with predicted values
    def check_accuracy(self):
        self.scaled_data = pd.DataFrame(self.scaler.transform(self.imputed_data), columns=self.data.columns) # maintain column names

        # One-Class SVM for outlier detection
        outliers_ocsvm = self.ocsvm.fit_predict(self.scaled_data)
        outliers = np.where(outliers_ocsvm == -1)[0] # finds outliers

        for index in outliers:
            row_index, col_index = divmod(index, self.num_col)
            if pd.isna(self.data.iloc[row_index, col_index]) or self.data.iloc[row_index, col_index] == -9999: # re-impute if it's still NaN or -9999
                self.data.iloc[row_index, col_index] = self.imputed_data.iloc[row_index, col_index] # replaces value of outlier with predicted value 

        # Transform scaled data back to original scale
        self.data[:] = self.scaler.inverse_transform(self.scaled_data)

    # Calculate reward based on reduction of NaN, consistency with allowed data types, accuracy of data imputs (-- TO DO --)
    def calculate_reward(self):
        # Completeness: the less NaN values remaining, the higher the reward obtained
        if self.current_col >= self.num_col: 
            na_reward = -self.data.isna().sum().sum()
        else:
            current_na = self.data.iloc[:, self.current_col].isna().sum()
            if current_na > 0:
                na_reward = -current_na
            else:
                na_reward = 1
        
        # Accuracy: the less outliers, the smaller the penalty 
        self.imputed_data = pd.DataFrame(self.imputer.transform(self.data), columns=self.data.columns)
        self.scaled_data = pd.DataFrame(self.scaler.transform(self.imputed_data), columns=self.data.columns)
        outliers_ocsvm = self.ocsvm.predict(self.scaled_data) 
        ocsvm_count = np.sum(outliers_ocsvm == -1) # count remaining outliers after each episode and calculate penalty 
        if ocsvm_count == 0: # if no outliers then reward is 1, else reward is negative sum of remaining outliers
            outlier_reward = 1
        elif ocsvm_count != 0:
            outlier_reward = - ocsvm_count
        
        # Total reward 
        full_reward = na_reward + outlier_reward
        return full_reward


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
        best_next_action = np.argmax(self.q_table[next_state_index]) # Greedy strategy for future state
        td_prediction = reward + self.gamma * self.q_table[next_state_index][best_next_action] * (1 - done) # TD prediction formula: reward + discounting rate * max Q(S(t+1), A), (1 - done) to ignore terminal rewards that are irrelevant
        td_error = td_prediction - self.q_table[state_index][action] # TD error formula: TD prediction - current Q(S, A) 
        self.q_table[state_index][action] += self.lr * td_error[action] # TD learning formula: Q(S, A) + TD error * learning rate (alpha)

        # decrease exploration to encourage exploitation with each episode 
        if done: 
            self.epsilon = max(self.min_epsilon, self.epsilon * self.gamma) # choose max value between epsilon * discounting rate and 0.01 (convergence towards exploitation)

    def update_state(self, state):
        self.current_state = self.state_to_index(state) # state index updated 

if __name__ == "__main__":
    env = Environment(train_data)
    agent = Agent(n_state=env.observation_space.shape[0], n_action=env.action_space.n)
    
    episodes = 20
    for episode in range(episodes):
        state = env.reset()
        reward = env.calculate_reward()
        agent.update_state(state)
        total_reward = 0
        
        while True:
            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done) # SARS -> Q learning 
            state = next_state
            total_reward += reward
            print(env.data)

            print(f"Episode: {episode + 1}, Iteration: {env.iteration}, Column: {env.current_col}, Reward: {reward}")

            if done: 
                print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Exploration Rate: {agent.epsilon}")
                print(env.data)
                break

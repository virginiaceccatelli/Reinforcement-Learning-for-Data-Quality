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

file_path = r"C:\Users\4967\OneDrive - Wavestone Germany Group\Dokumente\problem_klein.xlsx"
data = pd.read_excel(file_path)

# Data formatting (NA to NaN) 
data.replace(["NA", "", "null", "Na", "N/A", "na"], np.nan, inplace=True)
print(data.head())

# Encoding of categorical variables as numeric: decode after ML imputation to get original dataset 
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    if data[column].nunique() <= 20: # if there are 20 different entries it is treated as continuous text data, otherwise it is encoded as categorical
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column].astype(str))
        label_encoders[column] = le

# Encode continuous text data: TF-IDF (Term Frequency-Inverse Document Frequency) - Converts text into a matrix of TF-IDF features
vectorizer = {}
text_column = text_columns = [column for column in data.select_dtypes(include=['object']).columns if data[column].nunique() > 20]

for column in text_column:
    text_vectors = vectorizer.fit_transform(data[text_column].astype(str).fillna('')).toarray() # Numpy array: each row corresponds to a row from the original DataFrame and each column corresponds to a TF-IDF feature
    # The list comprehension iterates over the range of column indices and creates unique names by concatenating the original column name and the index
    text_df = pd.DataFrame(text_vectors, columns=[f"{column}_{i}" for i in range(text_vectors.shape[1])]) # Each row in text_vectors corresponds to a row in the new DataFrame
    data = pd.concat([data.drop(columns=[text_column]), text_df], axis=1)
    vectorizers[column] = vectorizer

# Train 75%, Test 25%: after training, use unseen data in test subset to estimate algorithm validity 
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

        # Action space describes the possible values that actions can take (finite) 
        self.action_space = spaces.Discrete(self.num_row)

        # Observation space (Box) describes valid values that observations can take for consistent state representation  
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_row, self.num_col), dtype=np.float32) 

    # To reset dataset back to original values after each episode of training 
    def reset(self):
        self.data = self.original_data.copy() 
        self.iteration = 0 
        self.current_col = 0 
        return self.data.values.flatten() 

    # To iterate over each row in every column and apply impute_value and calculate_reward 
    def step(self, action): 
        row_i = action 
        column_i = self.current_col
        if pd.isna(self.data.iloc[row_i, column_i]):
            self.data.iloc[row_i, column_i] = self.impute_value(row_i, column_i)
        self.iteration += 1

        # Move to next column 
        if self.iteration >= self.num_row:
            self.iteration = 0 
            self.current_col += 1
        
        finished = self.current_col >= self.num_col 
        reward = self.calculate_reward()
        if finished: 
            self.finish = True 
        return self.data.values.flatten(), reward, finished, {}

    # To impute missing values based on ML imputation method 'Iterative Imputer' using 'Random Forest Regressor'
    def impute_value(self, row_i, column_i):
        # Iterative Imputer with Random Forest Regressor as base model 
        target_col = self.data.columns[column_i]

        imputer = IterativeImputer(estimator=RandomForestRegressor(), max_iter=10, random_state=0)
        imputed_data = imputer.fit_transform(self.data)
        return imputed_data[row_i, column_i]

    # To calculate reward based on reduction of NaN, consistency with allowed data types, accuracy of data imputs (TO DO)
    def calculate_reward(self):
        # Completeness (to refine): the less NaN values, the higher the reward obtained 
        na_reduction = -self.data.isna().sum().sum()
        
        # Validity (to refine): if 
        consistency_reward = 0
        for column in self.data.columns:
            if self.data[column].dtype in [np.float64, np.int64] and (self.data[column] < 0).any():
                consistency_reward -= 10  
        
        # Accuracy (to do) 
        accuracy_reward = 0
        
        # Total reward
        total_reward = na_reduction + consistency_reward + accuracy_reward
        return total_reward


class Agent: 
    def __init__(self, n_state, n_action): 
        self.epsilon = 0.85 # exploration 
        self.min_epsilon = 0.01 # min exploration as exploitation becomes more important
        self.lr = 0.2 # learning rate: adjust Q values to converge towards optimal strategy 
        self.gamma = 0.99 # discounting rate (value of future rewards)
        self.n_state = n_state 
        self.n_action = n_action
        self.q_table = np.zeros((n_state, n_action))

    def state_to_index(self, state):
        return hash(tuple(state)) % self.n_state

    # Q learning: Temporal Difference Algorithm (Policy Evaluation)
    def take_action(self, state):
        if np.random.rand() > self.epsilon: # Epsilon greedy strategy for current state
            return np.random.choice(self.n_action) # Exploration
        return np.argmax(self.q_table[self.current_state])

    def learn(self, state, next_state, action, reward, finished): 
        state_index = self.state_to_index(state)
        best_next_action = np.argmax(self.q_table[next_state]) # Greedy strategy for future state
        td_prediction = reward + self.gamma * self.q_table[next_state, best_next_action] * (1 - finished)
        td_error = td_prediction - self.q_table[state_index, action]
        td_error_scalar = np.mean(td_error)
        self.q_table[state_index, action] += (self.lr * td_error_scalar)

        # decrease exploration to encourage exploitation 
        if finished: 
            self.epsilon = max(self.min_epsilon, self.epsilon * self.gamma)

    def update_state(self, state):
        self.current_state = self.state_to_index(state)


if __name__ == "__main__":
    env = SampleEnvironment(train_data)
    agent = Agent(n_state=env.observation_space.shape[0], n_action=env.action_space.n)
    
    episodes = 100
    for episode in range(episodes):
        state = env.reset()
        reward = env.calculate_reward()
        agent.update_state(state)
        total_reward = 0
        
        while True:
            action = agent.take_action(state)
            next_state, reward, finished, _ = env.step(action)
            agent.learn(state, action, reward, next_state, finished)
            state = next_state
            total_reward += reward
            
            if finished:
                print(f"Episode: {episode+1}, Total Reward: {total_reward}, Exploration Rate: {agent.epsilon}")
                break 

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

file_path = r"C:\Users\4967\OneDrive - Wavestone Germany Group\Dokumente\problem_klein.xlsx"
data = pd.read_excel(file_path)

# Data formatting (NA to NaN) 
data.replace(["NA", "", "null", "Na", "N/A", "na"], np.nan, inplace=True)
print(data.head())

# Encoding of categorical variables as numeric: decode after ML imputation to get original dataset 
label_encoders = {}
categorical_columns = []

for column in data.select_dtypes(include=['object']).columns:
    if data[column].nunique() <= 20: 
        series = data[column]
        label_encoder = LabelEncoder()
        data[column] = pd.Series(
            label_encoder.fit_transform(series[series.notnull()]),
            index=series[series.notnull()].index
        )
        label_encoders[column] = label_encoder
        categorical_columns.append(column)

# Train 75%, Test 25%: after training, use unseen data in test subset to estimate algorithm accuracy and validity 
train_data, test_data = train_test_split(data, test_size=0.25, random_state=0)
print(data.head())


''' NOT NEEDED FOR NOW '''

# Encode continuous text data: TF-IDF (Term Frequency-Inverse Document Frequency) - Converts text into a matrix of TF-IDF features (not tested yet)
vectorizer = {}
text_column = text_columns = [column for column in data.select_dtypes(include=['object']).columns if data[column].nunique() > 20]

for column in text_column:
    text_vectors = vectorizer.fit_transform(data[text_column].astype(str).fillna('')).toarray() # Numpy array: each row corresponds to a row from the original DataFrame and each column corresponds to a TF-IDF feature
    # The list comprehension iterates over the range of column indices and creates unique names by concatenating the original column name and the index
    text_df = pd.DataFrame(text_vectors, columns=[f"{column}_{i}" for i in range(text_vectors.shape[1])]) # Each row in text_vectors corresponds to a row in the new DataFrame
    data = pd.concat([data.drop(columns=[text_column]), text_df], axis=1) # Replaces old text data with new vector 
    vectorizers[column] = vectorizer


# Environment and Actions for RL
class Environment(gym.Env): # OpenAI Gym Environment Inheritance 
    def __init__(self, data, categorical_columns):
        super(Environment, self).__init__()
        self.data = data
        self.original_data = self.data.copy()
        self.iteration = 0
        self.finish = False
        self.current_col = 0
        self.num_col = len(self.data.columns)
        self.num_row = len(self.data)
        self.previous_na = self.data.isna().sum().sum()
        self.categorical_columns = categorical_columns

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
            self.data.iloc[row_i, column_i] = self.impute_value(row_i, column_i) 
        self.check_accuracy()
        self.iteration += 1

        # -- TO DO -- check if data entry is valid, if not apply functions   
        
        # Move to next column 
        if self.iteration >= self.num_row:
            self.iteration = 0 
            self.current_col += 1
        
        done = self.current_col >= self.num_col 
        reward = self.calculate_reward()
        if done: 
            self.finish = True 
        return self.data.values.flatten(), reward, done, {}
        
    # COMPLETENESS: Impute missing values based on ML imputation method 'Iterative Imputer' using 'Random Forest Regressor' 
    def impute_value(self, row_i, column_i):        
        target_col = data.columns[column_i]

        if target_col in self.categorical_columns:
            imputed_data = self.imputer.fit_transform(self.data)
            imputed_value = np.round(imputed_data[row_i, column_i])
        else:
            imputed_data = self.imputer.fit_transform(self.data)
            imputed_value = imputed_data[row_i, column_i]
            
        return imputed_value
        
    # -- TO DO -- function for validity; replacement of impossible inputs 

    # ACCURACY: To replace inaccurate values (that are disproportionate/ outliers compared to the rest of the data) with predicted values
    def check_accuracy(self):
        # Impute missing values first and then standardise data 
        self.imputed_data = self.imputer.transform(self.data) # impute using the trained imputer without fitting again
        self.scaled_data = pd.DataFrame(self.scaler.transform(self.imputed_data), columns=self.data.columns) # Maintain column names

        # One-Class SVM for outlier detection
        outliers_ocsvm = self.ocsvm.fit_predict(self.scaled_data)
        outliers = np.where(outliers_ocsvm == -1)[0] # finds outliers

        for index in outliers: # finds indices for outliers 
            row_index = index // self.num_col
            col_index = index % self.num_col

            imputed_value = self.imputed_data[row_index, col_index] 
            self.data.iloc[row_index, col_index] = imputed_value # replaces value of outlier with predicted value 

    # Calculate reward based on reduction of NaN, consistency with allowed data types, accuracy of data imputs (-- TO DO --)
    def calculate_reward(self):
        # Completeness: the less NaN values remaining, the higher the reward obtained
        current_na = self.data.isna().sum().sum()
        na_reduction = self.previous_na - current_na # positive reward for reduction of NaN 
        self.previous_na = current_na # previous_na constantly updated to encourage reduction of multiple NaN values per episode 
        
        # Validity: check if imputed values are of allowed datatype in column  
        #imputed_values = self.check_format(row_i, column_i) # TO GET FROM FUNCTION  
        #for column in data.columns:
        #    column_type = data[column].dtype
        #    for value in data[column]:
        #        if value in imputed_values:
        #            if isinstance(value, column_type): 
        #                validity_reward = -10
        #            else:
        #                validity_reward = -1
        
        # Accuracy: the less outliers, the smaller the penalty 
        self.imputed_data = self.imputer.fit_transform(self.data)
        self.scaled_data = self.scaler.transform(self.imputed_data)
        outliers_ocsvm = self.ocsvm.predict(self.scaled_data)
        ocsvm_count = np.sum(outliers_ocsvm == -1) # count remaining outliers after each episode and calculate penalty (negative sum of remaining outliers) 
        if ocsvm_count == 0: # if no outliers then reward is 1, else reward is negative sum 
            outlier_penalty = 1
        else:
            outlier_penalty = - ocsvm_count 
        
        # Total reward 
        total_reward = na_reduction + outlier_penalty
        return total_reward


class Agent: 
    def __init__(self, n_state, n_action): 
        self.epsilon = 0.75 # exploration 
        self.min_epsilon = 0.01 # min exploration as exploitation becomes more important
        self.lr = 0.2 # learning rate: adjust Q values to converge towards optimal strategy 
        self.gamma = 0.99 # discounting rate (value of future rewards)
        self.n_state = n_state 
        self.n_action = n_action
        self.q_table = np.zeros((n_state, n_action)) # initial Q table for learning (all zeros because no value is updated yet) 

    # state not iterable in learn(); state_index needed
    def state_to_index(self, state): 
        return hash(tuple(state)) % self.n_state 

    # Q learning Algorithm
    def take_action(self, state):
        state_index = self.state_to_index(state)
        if np.random.rand() > self.epsilon: # Epsilon greedy strategy for current state
            return np.random.choice(self.n_action) # Exploration
        return np.argmax(self.q_table[state_index])

    def learn(self, state, next_state, action, reward, done): 
        state_index = self.state_to_index(state)
        action_converted = action.astype(int)

        # Temporal Difference Learning for Policy Evaluation 
        best_next_action = np.argmax(self.q_table[next_state]) # Greedy strategy for future state
        td_prediction = reward + self.gamma * self.q_table[next_state, best_next_action] * (1 - done) # (1 - done) to ignore terminal rewards that are not relevant: 1 - True = 0
        td_error = abs(td_prediction - self.q_table[state_index, action_converted])
        td_error_scalar = np.mean(td_error)
        self.q_table[state_index, action_converted] += (self.lr * td_error_scalar)

        # decrease exploration to encourage exploitation 
        if done: 
            self.epsilon = max(self.min_epsilon, self.epsilon * self.gamma)

    def update_state(self, state):
        self.current_state = self.state_to_index(state)


if __name__ == "__main__":
    env = Environment(train_data, categorical_columns)
    agent = Agent(n_state=env.observation_space.shape[0], n_action=env.action_space.n)
    
    episodes = 50
    for episode in range(episodes):
        state = env.reset()
        reward = env.calculate_reward()
        agent.update_state(state)
        total_reward = 0
        
        while True:
            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            print(env.scaled_data)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            print(f"Episode: {episode + 1}, Iteration: {env.iteration}, Column: {env.current_col}, Reward: {reward}")

            if done: 
                print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Exploration Rate: {agent.epsilon}")
                break

# Reinforcement Learning for Data Quality Automation

This project aims to automate and improve data-cleaning procedures using a Reinforcement Learning Algorithm. The algorithm processes a dataset, refining it in terms of data completeness, validity, and accuracy:

NA values are replaced with optimal values based on Machine Learning techniques (Random Forest Regressor and K-NN).
Invalid values (such as out-of-range values like incorrect ages or invalid data types) are corrected using the same methods.
Outliers are identified, and the dataset's overall accuracy, related to the number of detected outliers (non-normal instances), is assessed. This is accomplished through a Machine Learning Algorithm: One Class Support Vector Machine.

## Algorithms used: 
#### Reinforcement Learning Algorithms: 
Q Learning: Q Learning utilizes Q-values to estimate the expected utility of taking a specific action in a given state, updating these values with the Bellman equation. This equation identifies the optimal action to maximize rewards. The key advantage of Q-learning is its simplicity and ability to learn optimal policies, while its drawbacks include potential inefficiency in large state spaces and slow convergence.

Temporal Difference Learning: TD learning updates value estimates by calculating the difference between successive predictions, determining the best next action to achieve the highest reward. Its primary advantage is its ability to learn directly from raw experience, enabling faster learning and more efficient data usage. However, its disadvantages include potential instability in predictions.

#### Value Imputation Algorithms: 
Random Forest Regressor: Random forest is a machine learning algorithm that generates an ensemble of multiple decision trees to produce a more accurate prediction or result. Its advantages include high predictive accuracy, speed, resistance to overfitting, capability to handle large datasets, and assess variable importance.

K Nearest Neighbors: The K-NN algorithm identifies the K closest neighbors to a given data point based on a distance metric, such as Euclidean distance. The class or value of the data point is then determined by the majority vote or average of the K neighbors. This method allows the algorithm to adapt to different patterns and make predictions based on the local structure of the data.

#### Outlier Detection Algorithm: 
One Class Support Vector Machine: One-Class Support Vector Machine is a Machine Learning Algorithm designed for outlier, anomaly, or novelty detection. It solves convex optimization problems of maximizing the distance between the two classes (normality and outliers in the dataset). While its hyperparameters are easily adjustable, the algorithm risks wrongful classification and prediction, as well as potential inefficiency in large state spaces. 


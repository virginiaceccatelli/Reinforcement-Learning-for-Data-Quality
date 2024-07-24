# Reinforcement-Learning-for-Data-Quality

This project focuses on automating and enhancing data-cleaning processes through a Reinforcement Learning Algorithm. 
The algorithm takes a dataset as input and curates it in terms of data completeness, data validity and data accuracy: 
  - NA values are replaced with best-fit values according to Machine Learning processes (Random Forest Regressor and K-NN)
  - Invalid values (out-of-bound invalid values such as incorrect age and invalid datatypes) are replaced with the same process
  - Outliers are detected and the overall accuracy of the dataset, related to the number of detected outliers (non-normal instances), is highlighted. This is done through a Machine Learning Algorithm: One Class Support Vector Machine 

## Algorithms used: 
Reinforcement Learning Algorithms: 
Q Learning: Q Learning uses Q-values, which estimate the expected utility of taking a given action in a given state and updates these values using the Bellman equation. This equation calculates the best action to take to achieve the most rewards. The main advantage of Q-learning is its simplicity and ability to learn optimal policies, while its disadvantages include potential inefficiency in large state spaces and slow convergence.
Temporal Difference Learning: TD learning updates value estimates based on the difference between successive predictions – it therefore calculates what the best next action would be to achieve the highest reward. Its main advantage is the ability to learn directly from raw experience, allowing for faster learning and more efficient use of data. However, its disadvantages include potential instability in its predictions.

Value Imputation Algorithms: 
Random Forest Regressor: Random forest is a machine learning algorithm that creates an ensemble of multiple decision trees to reach a singular, more accurate prediction or result. Its benefits include high predictive accuracy, speed, resistance to overfitting, ability to handle large datasets and assess variable importance. 
K Nearest Neighbours: The K-NN algorithm works by finding the K nearest neighbors to a given data point based on a distance metric, such as Euclidean distance. The class or value of the data point is then determined by the majority vote or average of the K neighbors. This approach allows the algorithm to adapt to different patterns and make predictions based on the local structure of the data. 

Outlier Detection Algorithm: 
One Class Support Vector Machine: One-Class Support Vector Machine is Machine Learning Algorithm that is designed for outlier, anomaly, or novelty detection. It solves convex optimization problems of maximizing the distance between the two classes (normality and outliers in the dataset). While its hyperparameters are easily adjustable, the algorithm risks wrongful classification and prediction, as well as potential inefficiency in large state spaces. 


# Reinforcement-Learning-for-Data-Quality

This project focuses on automating and enhancing data-cleaning processes through a Reinforcement Learning Algorithm. 
The algorithm takes a dataset as input and curates it in terms of data completeness, data validity and data accuracy: 
  - NA values are replaced with best-fit values according to Machine Learning processes (Random Forest Regressor and K-NN)
  - Invalid values (out-of-bound invalid values such as incorrect age and invalid datatypes) are replaced with the same process
  - Outliers are detected and the overall accuracy of the dataset, related to the number of detected outliers (non-normal instances), is highlighted. This is done through a Machine Learning Algorithm: One Class Support Vector Machine 

Reinforcement Learning Algorithms: Q Learning with Temporal Difference Learning 

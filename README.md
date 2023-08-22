# Hive-Mortality-Prediction
A machine-learning program to predict the seasonal mortality of honeybee colonies

This README contains information on how to run the code associated with the project described in Hive_mortality_prediction.pdf
Authors: Timothy Lam (tlam04@uoguelph.ca), Samantha Share (sshare@uoguelph.ca)

To run the full pipeline: 
python3 Master_script.py


This master script calls several different scripts.
Below is a description of each script in the order they are called by Master_script.py:
1. Add_weather.py: Adds a mean temperature column to the EPILOBEE dataset
2. data_preprocessing.py: processes the data, removing NA's and missing values and encodes categoricals
3. Trim_data.py: An optional step, reduces the size of EPILOBEE dataset to reduce computational complexity.
4. feature_selection.py: Performs feature selection with SelectKBest based on F-value scores.
5. DT_model_Optimized.py: Trains and tests decision tree based models to predict colony mortality. 
6. GBR_model_Optimized.py: Trains and tests gradient boosting machine based models to predict colony mortality. 
7. RF_model_Optimized.py: Trains and tests random forest based models to predict colony mortality. 
8. SVM_model_Optimized.py: Trains and tests support vector machine based models to predict colony mortality. 

Running this pipeline will result in several .txt files corresponding to the selected features from recursive features engineering for each model.
In addition, .sav files will be produced corresponding to each model and .png files for visualizations of model performance. 


To run the Colony Mortality Prediction Program: 
1. Navigate to the subdirectory titled: "Colony Mortality Prediction Program"
2. Enter: python3 Final_program.py
3. Respond to queries to generate a prediction of colony mortality for your apirary


Thanks for using our program! If you have any questions or comments please direct them to tlam04@uoguelph.ca or sshare@uoguelph.ca

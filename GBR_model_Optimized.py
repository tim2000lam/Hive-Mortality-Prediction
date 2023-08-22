#Pipeline for gradient boosting regression model on EPILOBEE dataset
#Timothy Lam (tlam04@uoguelph.ca), Samantha Share (sshare@uoguelph.ca)
#How to run: python3, GBR_model_Optimized.py

from pandas import read_csv 
from pandas import DataFrame
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.svm import SVR
from numpy import arange
import random
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_percentage_error
import joblib
from sklearn.model_selection import learning_curve
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor


# set font sizes for plotting
SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

pyplot.rc('font', size=SMALL_SIZE)          # controls default text sizes
pyplot.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
pyplot.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
pyplot.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
pyplot.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
pyplot.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
pyplot.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


#Read in CSV
dataset = read_csv("bee_data_preprocessed.csv")


# separate data into training/validation and testing datasets
array = dataset.values
X = array[:,1:] # input features are in all columns except the first one
#print(X)
y = array[:,0] # target variable is in the first column
#print(y)
random.seed(42)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)


#Define list for storing model
models = []
models.append(('GBM', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)))  #NOTE: DOING IT IN A LIST ISN'T NECESSARY SINCE WERE DOING ONE MODEL PER SCRIPT BUT I DIDNT WANT TO HAVE TO CHANGE AS MUCH FROM DANS EXAMPLE

#Define a dictionary for storing number of features and a boolean array of the features selected
features_dict = {}
number_of_features = [5,10,20,30, 37] 

print('\nGBR Recursive Feature Elimination')
print('--------------------------')

#Perform recursive feature engineering with varying numbers of features
for i in number_of_features:
    for name, model in models:
        model.fit(X_train, Y_train)
        rfe = RFE(model, n_features_to_select = i)
        fit = rfe.fit(X,y)
        X_rfe = fit.transform(X)
        features_dict[i] = fit.support_  #Append boolean array of selected features to features_dict

#Define dict for storing the number of features (key) training score for each subset of features (For making boxplots on pre-tuned models)
score_by_feature = {}

# create empty dictionary to store learning curve data
learning_curve_data = {}

###Model Training
print('\nGBR Model evaluation - training')
print('--------------------------')
for i in number_of_features:
    selected_features = features_dict[i]
    X_train_selected = X_train[:, selected_features] #Trim the training set to include only the features that were selected for this iteration of i
    X_test_selected = X_test[:, selected_features]

    # evaluate models
    print("Number of features: " +str(i))
    results = []
    for name, model in models:
        kfold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        cv_results = cross_val_score(model, X_train_selected, Y_train, cv=kfold, scoring='neg_mean_absolute_error', n_jobs=-1, error_score='raise')
        results.append(cv_results)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
        score_by_feature[i] = results

    # Generate learning curve and plot
    train_sizes, train_scores, val_scores = learning_curve(model, X_train_selected, Y_train, cv=kfold, scoring='neg_mean_absolute_error', n_jobs=-1)
    learning_curve_data[i] = {'train_sizes': train_sizes, 'train_scores': train_scores, 'val_scores': val_scores}
    

#Convert scores for training to list
labels = list(score_by_feature.keys())
values = []
for score in list(score_by_feature.values()):
    for x in score:
        x = x.tolist()
        values.append(x)

#Plot training scores before optimization
pyplot.boxplot(values, labels=labels)
pyplot.xlabel("Number of predictor variables")
pyplot.ylabel("Negative Mean Absolute Error")
pyplot.title("Gradient Boosting Machine: Number of predictor \n variables comparison, before optimization")

pyplot.savefig('GBM_before.png')

pyplot.clf()



# plot learning curves
for i in number_of_features:
    data = learning_curve_data[i]
    train_sizes = data['train_sizes']
    train_scores = data['train_scores']
    val_scores = data['val_scores']
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = -np.std(train_scores, axis=1)
    val_scores_mean = -np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    fig, ax = pyplot.subplots()
    ax.set_title("Learning Curve GBM, Number of features: " + str(i)) 
    ax.set_xlabel("Training Examples")
    ax.set_ylabel("Score")

    ax.grid()
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1,
                    color="r")
    ax.fill_between(train_sizes, val_scores_mean - val_scores_std,
                    val_scores_mean + val_scores_std, alpha=0.1, color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
            label="Training score")
    ax.plot(train_sizes, val_scores_mean, 'o-', color="g",
            label="Cross-validation score")
    ax.legend(loc="best")

    fig_name = "GBM_LC_before_+" +str(i) +".png"
    
    pyplot.savefig(fig_name)

pyplot.clf()

optimized_models = {}
score_by_feature_optimized = {}


param_grid = {'learning_rate': [0.1, 0.05, 0.01], 'max_depth': [3, 5, 7], 'n_estimators': [100, 500, 1000], 'subsample': [0.8, 0.9, 1.0],'min_samples_split': [2, 5, 10]}

#Hyperparameter optimization
print('\nGBR Hyperparameter Optimization')
print('--------------------------')

for i in number_of_features: #This for loop loops through each  subset of features
    selected_features = features_dict[i]
    X_train_selected = X_train[:, selected_features] #Trim the training set to include only the features that were selected for this iteration of i
    X_test_selected = X_test[:, selected_features] #Trim the testing set to include only the features that were selected for this iteration of i

    #run the search on the model from above
    search=RandomizedSearchCV(model,param_distributions=param_grid,scoring='neg_mean_squared_error',cv=5)

    result = search.fit(X_train_selected, Y_train)
    print("Number of features: " +str(i))
    print('Best Hyperparameters: %s' % result.best_params_)

    #Re-train the model with the best parameters:
    # Get the best estimator from the grid search
    best_model = result.best_estimator_

    #Define cross validation approach
    kfold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    cv_results = cross_val_score(best_model, X_train_selected, Y_train, cv=kfold, scoring='neg_mean_absolute_error', n_jobs=-1, error_score='raise')
    #extract results from training of optimized models 
    results = []
    results.append(cv_results)
    score_by_feature_optimized[i] = results
    optimized_models[i] = best_model

    #Print results after hyperparameter optimization
    print('%s: %f (%f)' % (best_model, cv_results.mean(), cv_results.std()))
    
#Convert scores for training to list (For plotting training scores from optimized models 
labels = []
values = []
labels = list(score_by_feature_optimized.keys())
values = []
for score in list(score_by_feature_optimized.values()):
    for x in score:
        x = x.tolist()
        values.append(x)

#Plot training scores from optimized models
pyplot.boxplot(values, labels=labels)
pyplot.xlabel("Number of predictor variables")
pyplot.ylabel("Negative Mean Absolute Error")
pyplot.title("GBM: Number of predictor variables \n comparison, after optimization")

pyplot.savefig('GBM_after.png')

pyplot.clf()

###Save optimized models
for i in number_of_features:
    selected_features = features_dict[i]
    X_train_selected = X_train[:, selected_features]

    #fit and save optimized models
    model = optimized_models[i]
    model.fit(X_train_selected, Y_train)
    filename = str(i) + 'GBM_model.sav'
    joblib.dump(model, filename)


mse_by_subset = {} #Dictionary to store MSE for each number of features  
###Testing with optimized models for each subset
print('\nModel Testing - optimized')
print('--------------------------')
for i in number_of_features:
    selected_features = features_dict[i]
    X_train_selected = X_train[:, selected_features] #Trim the training set to include only the features that were selected for this iteration of i
    X_test_selected = X_test[:, selected_features]#Trim the test set to include only the features that were selected for this iteration of i

    best_model = optimized_models[i]
    
    best_model.fit(X_train_selected, Y_train)
    #Predict on the unseen test set
    y_pred = best_model.predict(X_test_selected)

    #Calculate the mean squared error
    mse = mean_squared_error(Y_test, y_pred)
    print("Number of features: " +str(i))
    print('Test MSE: %.3f' % mse)
    mse_by_subset[i] = mse

    #Calculate learning curves
    train_sizes, train_scores, val_scores = learning_curve(best_model, X_train_selected, Y_train, cv=5, 
                                                            scoring='neg_mean_absolute_error', 
                                                            n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = -np.std(train_scores, axis=1)
    val_scores_mean = -np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    #Plot learning curves
    fig, ax = pyplot.subplots()
    ax.set_title("Learning Curve GBM Post Optimization,\n Number of features: " + str(i))
    ax.set_xlabel("Training Examples")
    ax.set_ylabel("Score")
    ax.grid()
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    ax.fill_between(train_sizes, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, alpha=0.1, color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Cross-validation score")
    ax.legend(loc="best")

    fig_name = "GBM_LC_after_+" +str(i) +".png"
    
    pyplot.savefig(fig_name)


pyplot.clf()

###For each sized subset, save the predictor variables that make up each subset 
#Extract columns for predictor variables:
cols = dataset.columns[1:].tolist()

#Convert X_train array to Pandas DataFrame
X_train_df = DataFrame(X_train, columns=cols)


#Extract the names of the selected features for each model and the MSE for those features 
for i in number_of_features:
    selected_features = features_dict[i]
    features = X_train_df.columns[selected_features].tolist()
    filename = str(i) + name + '_Pvariables.txt'
    with open(filename, 'w') as f:
        f.write("%s\n" % mse_by_subset[i]) #Save MSE 
        for variable in features:
            f.write("%s\n" % variable) #Save features 





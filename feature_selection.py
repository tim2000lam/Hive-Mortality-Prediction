import pandas as pd 
from numpy import set_printoptions
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
import matplotlib.pyplot as plt

#load the data file in
bee_data = pd.read_csv('bee_data_preprocessed.csv')

      
#Separate input and output variable
varray=bee_data.values
X = varray[:,1:40]
Y = varray[:,0]

#Select k=10 best features
test = SelectKBest(score_func=f_classif, k=10)
fit = test.fit(X,Y)

#summarize scores
set_printoptions(precision=10)
print("Feature scores:", fit.scores_)

#summarize selected features
features = fit.transform(X)
selected_cols = test.get_support(indices=True)
selected_features_names = bee_data.columns[1:40][selected_cols]
selected_features = X[:, selected_cols]
print("Selected features: \n", pd.DataFrame(selected_features, columns=selected_features_names))

print(selected_features_names)

# Create a bar chart of the feature scores
plt.bar(selected_features_names, fit.scores_[selected_cols])
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Score')
plt.title('Feature Scores')
plt.savefig("feature_scores.png")


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import seaborn as sns               # to create scatter plots
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV        # tool for tunning hyperparameters of ML models
from sklearn.ensemble import BaggingRegressor


# 1. Get database and create a dataframe
url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data")    # link that has the database file
abalone = pd.read_csv(url, header=None)

abalone.columns = ["Sex",
                    "Length",
                    "Diameter",
                    "Height",
                    "Whole weight",
                    "Shucked weight",
                    "Viscera weight",
                    "Shell weight",
                    "Rings",
                    ]

abalone = abalone.drop("Sex", axis=1)       # we don't need Sex as we are looking for physical measurements
# print(abalone.head())

# 2. Create a chart to understand better the distribution of the target variable
abalone["Rings"].hist(bins=15)              # Rings is the target variable
# plt.show()                                # uncomment if you want to see the chart

# 3. Create a correlation matrix to see the correltions of independent variables and target variable
correlation_matrix = abalone.corr()
correlation_matrix_rings = correlation_matrix["Rings"]
# print(correlation_matrix_rings)             # the closer they are to 1, the more correlation there is

# 4. Transform data to vectors
X = abalone.drop("Rings", axis=1)           # independent variables
X = X.values
y = abalone["Rings"]                        # dependent variable
y = y.values

# 5. Get new data point
new_data_point = np.array([
    0.569552,
    0.446407,
    0.154437,
    1.016849,
    0.439051,
    0.222526,
    0.291208,
    ])

# 6. Compute the distances between this new data point and each of the previous data points

distances = np.linalg.norm(X - new_data_point, axis=1)      # vector of distances

# 7. Find k closest neighbors

k = 3                           # you can choose whatever k you like
nearest_neighbor_ids = distances.argsort()[:k]      # closest neighbors to your new data point
# print(nearest_neighbor_ids)

# 8. Convert neighbors in an estimation

# 8.1 Find the ground truths for those k neighbors

nearest_neighbor_rings = y[nearest_neighbor_ids]
# print(nearest_neighbor_rings)

# 8.2 Combine neighbors into a prediction for your new data point

prediction = nearest_neighbor_rings.mean()      # because this is a regression (target variable is numeric)
# # if this was a classification problem, you could use mode instead of mean:
# import scipy.stats
# class_neighbors = np.array(["A", "B", "B", "C"])
# scipy.stats.mode(class_neighbors)

# 9. Split train and test data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2,              # number of observations you want to put in the test data (training data will be 1- test_size)
    random_state = 12345                # allows you to obtain the same results every time the code is run
)

# 10. Fit a kNN model on the training data

knn_model = KNeighborsRegressor(n_neighbors = 3)        # n_neighbors is k

knn_model.fit(X_train, y_train)                         # this model is to make predictions on new data points

# 11. Evaluate the fit

train_preds = knn_model.predict(X_train)
mse_train = mean_squared_error(y_train, train_preds)
rmse_train = sqrt(mse_train)
rmse_train = round(rmse_train, ndigits = 2)
print("When we first evaluate the fit, on average, the error in the prediction is of "+str(rmse_train)+" years.")                    # the error on data that was known by the model

test_preds = knn_model.predict(X_test)
mse_test = mean_squared_error(y_test, test_preds)
rmse_test = sqrt(mse_test)
rmse_test = round(rmse_test, ndigits = 2)
diff1 = round(rmse_test - rmse_train, ndigits = 2)
print("When we do the same thing but with test data, the error is of "+str(rmse_test)+" years.")       # the error on data that wasn't yet known by the model
print("The difference between the train and test data is "+str(diff1))

# 12. Check what the model is learning

cmap_predicted = sns.cubehelix_palette(as_cmap = True)
f, ax = plt.subplots()
points = ax.scatter(
    X_test[:,0],                # column Length
    X_test[:,1],                # column Diameter
    c = test_preds,             # predicted values used as a colorbar
    s = 50,                     # size ofthe points in the scatter plot
    cmap = cmap_predicted       # color map
)
f.colorbar(points)
# plt.show()                    # uncomment this to see the plot

cmap_test = sns.cubehelix_palette(as_cmap = True)
f, ax = plt.subplots()
points = ax.scatter(
    X_test[:,0],                # column Length
    X_test[:,1],                # column Diameter
    c = y_test,                 # predicted values used as a colorbar
    s = 50,                     # size ofthe points in the scatter plot
    cmap = cmap_test            # color map
)
f.colorbar(points)
# plt.show()                    # uncomment this to see the plot

# You can play with other dimensions. Here we used Length and Diameter only.

# 13. Optimize kNN model by finding best value of k

parameters = {"n_neighbors": range(1,50)}           # test the values for k from 1 to 50
gridsearch = GridSearchCV(KNeighborsRegressor(), parameters)
gridsearch.fit(X_train, y_train)
k_optimal = gridsearch.best_params_                 # optimal k value to use for the model
print("The optimal value for k is "+str(k_optimal))

# Using the optimal k value, you check again the prediction error for train and test
train_preds_grid = gridsearch.predict(X_train)
train_mse = mean_squared_error(y_train, train_preds_grid)
train_rmse = sqrt(train_mse)
train_rmse = round(train_rmse, ndigits =2)
test_preds_grid = gridsearch.predict(X_test)
test_mse = mean_squared_error(y_test, test_preds_grid)
test_rmse = sqrt(test_mse)
test_rmse = round(test_rmse, ndigits = 2)
diff2 = round(test_rmse - train_rmse, ndigits = 2)
print("When we use GridSearchCV, the error in the prediction is of "+str(test_rmse)+" years, and the difference with the train data is "+str(diff2))
# print(test_rmse)

# the results show that now the difference is much smaller than before
# the training error is worse than before, but the test error is better
# this means that your model fits less closely to the training data
# using GridSearchCV to find a value for k has reduced the problem of overfitting on the training data

# 14. Optimize the model even more by finding weighted average

# neighbors that are further away will less strongly influence the prediction
parameters = {
    "n_neighbors": range(1, 50),
    "weights": ["uniform", "distance"]
}
gridsearch = GridSearchCV(KNeighborsRegressor(), parameters)
gridsearch.fit(X_train, y_train)
gridsearch.best_params_
test_preds_grid = gridsearch.predict(X_test)
test_mse = mean_squared_error(y_test, test_preds_grid)
test_rmse = sqrt(test_mse)
test_rmse = round(test_rmse, ndigits = 3)
print("When we use GridSearchCV including weights, the error in the prediction is of "+str(test_rmse)+" years.")

# results show that the prediction error is slightly smaller than before

# 15. Optimize the model using bagging

# baggins is a method that takes a relatively straightforward ML model and fits a large number
# of those models with slight variations in each fit
# one model can be wrong from time to time, but the average of a hundred models should be wrong less often
best_k = gridsearch.best_params_["n_neighbors"]
best_weights = gridsearch.best_params_["weights"]
bagged_knn = KNeighborsRegressor(
    n_neighbors = best_k, weights = best_weights
)
bagging_model = BaggingRegressor(bagged_knn, n_estimators = 100).fit(X_train, y_train)

# now make a prediction and evaluate the RMSE to see if it improved
test_preds_grid = bagging_model.predict(X_test)
test_mse = mean_squared_error(y_test, test_preds_grid)
test_rmse = sqrt(test_mse)
test_rmse = round(test_rmse, ndigits = 3)
print("When we use Bagging and GridSearchCV, the error in the prediction is of "+str(test_rmse)+" years.")
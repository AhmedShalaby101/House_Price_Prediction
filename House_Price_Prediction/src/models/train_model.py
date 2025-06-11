import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import time
import pipreqs
## read dataset

X_train = pd.read_pickle('../../data/interim/model_X_train.pkl')
Y_train = pd.read_pickle('../../data/interim/model_Y_train.pkl')

## define training models
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(random_state=42),
    "SVR": SVR()

}

for name,model in models.items():
    
    print(f'traning data using {name} model')
    
#training the model 
    
    traind_model = model.fit(X_train,Y_train)

# check model performance
    y_train_pred = traind_model.predict(X_train)
    train_r2 = r2_score(Y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(Y_train, y_train_pred))
    print(f'train_r2 for {name}:{train_r2}')
    print(f'train_rmse for {name}:{train_rmse}')
#Cross Validation Evaluation
    scores = cross_val_score(traind_model,X_train,Y_train,
                        scoring='neg_mean_squared_error',cv=10)
    cv_rmse_mean = np.sqrt(-scores).mean()
    cv_rmse_std  = np.sqrt(-scores).std()
    print(f'cv for {name}: {cv_rmse_mean:.2f} ±  {cv_rmse_std:.2f}')
    print('='*50)
    
"""
Linear regression model:
 
has R² = 0.65 which is not bad and minimal overfitting

Decision tree:

has R² = 0 and high validation rsme means severe overfitting and very poor generalization

RandomForest:(best overall model)

has good  R² but slightly overfitted which will need Regularization

SVR :

A small negative R² on train data means it's still too constrained or missing signal.
which means the model is underfitting which also might need regularizartion 
"""

## Hyperparameter tunning 
models = {
    'SVR': SVR(),
    'RandomForest': RandomForestRegressor(random_state=42)
}
"""
    here we select hyper parameters to reduce overfit for RandomForest
    while choosing hyper parameters to reduce underfit for SVR
"""
params = {
    'SVR': {
        'C': [5,10,15,30],
        'kernel': ['linear'],
        'gamma': [ 0.001, 0.01,0.5],
        'epsilon': [0.01, 0.1, 0.5, 1],
    },
     'RandomForest': {
    'n_estimators': [80, 100, 120, 150],
    'max_depth': [8, 10, 12, 14],
    'max_features': [6, 8, 10],
    'min_samples_leaf': [2, 4, 6]
                    }
}



for model in models :
    print(f'hyper parameter tunning for {model} model')
    
    ##Tunning
    g_search  = GridSearchCV(estimator=models[model],param_grid=params[model],
                             cv=5,scoring='neg_root_mean_squared_error',
                             n_jobs=-1,verbose=2)
    start_time = time.time()
    
    g_search.fit(X_train,Y_train)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f'Best params: {g_search.best_params_}')
    
    # check model performance
    
    y_train_pred = g_search.predict(X_train)
    train_r2 = r2_score(Y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(Y_train, y_train_pred))
    print(f'train_r2 for {model}:{train_r2}')
    print(f'train_rmse for {model}:{train_rmse}')
    
    #Cross Validation
    
    best_index = g_search.best_index_
    cv_rmse_mean = -g_search.cv_results_['mean_test_score'][best_index]  
    cv_rmse_std = g_search.cv_results_['std_test_score'][best_index]

    print(f'cv for {model}: {cv_rmse_mean:.2f} ±  {cv_rmse_std:.2f}')
    print('='*10  + f'{elapsed_time:.1f}seconds' + '='*10)
    
    
tunned_Random_forest = g_search.best_estimator_

## check model performance on test data

X_test = pd.read_pickle('../../data/interim/model_X_test.pkl')
Y_test = pd.read_pickle('../../data/interim/model_Y_test.pkl')
y_test_pred = tunned_Random_forest.predict(X_test)
test_r2 = r2_score(Y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(Y_test, y_test_pred))
print(f'test_r2 for tunned_RF:{test_r2}')
print(f'test_rmse for tunned_RF:{test_rmse}')

"""
RMSE is just a point estmate so we need to be confident 
"""
## confidence interval

from scipy import stats
confidence = 0.95
sqrt_error = (Y_test - y_test_pred)**2
interval = np.sqrt(stats.t.interval(confidence,len(sqrt_error) - 1,
                                    loc = sqrt_error.mean(),
                                    scale= stats.sem(sqrt_error)))
print(interval)
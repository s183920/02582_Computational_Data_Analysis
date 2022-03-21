from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as lng
import scipy.io as io
from sklearn.ensemble import BaggingClassifier, AdaBoostRegressor
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from pathlib import Path
import pandas as pd
import seaborn as sns
from sklearn.metrics import make_scorer
from sklearn.model_selection import TimeSeriesSplit
# import pickle
import os
# from sklearn.externals import joblib
import joblib
import dill as pickle

##################################### DATA ########################################################
def get_data(data_format = "summary_stats"):
    """Gets the X and y var. Data format must be onehot or summary_stats"""
    data_file = "processed_data.csv"
    d = pd.read_csv(data_file)

    ## Vi bliver ikke bedømt på fly hvor der er nul passagerer
    d = d[d.LoadFactor != 0]

    ## Splits
    d.ScheduleTime = pd.to_datetime(d.ScheduleTime)
    ix_val = ((d.ScheduleTime.dt.year==2022) & (d.ScheduleTime.dt.month!=1))

    ## Fjerner udvalgte kolonner da de allerede er blevet encoded som sin/cos
    ## Eller hvis kolonnen kun indehodler identiske værdier
    X_cols = list(d.columns)
    X_cols.remove("LoadFactor")
    X_cols.remove("ScheduleTime")
    X_cols.remove('Hour')
    X_cols.remove('Date')
    X_cols.remove('QuarterEnd')
    X_cols.remove('MonthEnd')
    X_cols.remove('TimeToQuarterEnd')
    X_cols.remove('TimeToMonthEnd')
    X_cols.remove('Holiday')

    ## One-hot encoding
    d_one_hot = pd.get_dummies(d[X_cols],)

    ## Alternativ til One-hot encoding af features med mange levels
    ## Der udregnes middelværdi og precision (inverse variance) af LoadFactor, for hvert
    ## unikt level. Eks: For alle observationer flynummer SA201, udregnes der middelværdi og precision
    ## Disse summary metrics tilføjes som features
    for col in ["FlightNumber2","Airline","Sector","Destination","AircraftType"]:
        mu = d[~ix_val].groupby(col)["LoadFactor"].mean()
        prec = 1/d[~ix_val].groupby(col)["LoadFactor"].var()

        mu = d[col].map(mu)
        prec = d[col].map(prec)
        mu[mu.isna()|np.isinf(mu)] = 0
        prec[prec.isna()|np.isinf(prec)] = 0

        col_mean = "mean_"+col
        col_prec = "prec_"+col

        d_one_hot[col_mean] = mu
        d_one_hot[col_prec] = prec


    cols_mu = d_one_hot.columns.str.startswith("_mean")
    cols_prec = d_one_hot.columns.str.startswith("_prec")

    cols_flight_numbers = d_one_hot.columns.str.startswith("FlightNumber2")
    cols_airline = d_one_hot.columns.str.startswith("Airline")
    cols_sector = d_one_hot.columns.str.startswith("Sector")
    cols_destination = d_one_hot.columns.str.startswith("Destination")
    cols_aircraft = d_one_hot.columns.str.startswith("AircraftType")
    cols_one_hot = (cols_flight_numbers | cols_airline | cols_sector | cols_destination | cols_aircraft)


    ## Standardisering
    X = (d_one_hot-d_one_hot.mean(axis=0))/d_one_hot.std(axis=0)


    ## To datamatrixer: En med alt one-hot encoded og en anden hvor mean+precision benyttes
    ## for udvalgte features.
    X_full_cols = d_one_hot.columns[~(cols_mu | cols_prec)]
    X_small_cols = d_one_hot.columns[~cols_one_hot]

    X_full = X[X_full_cols].to_numpy()
    X_small = X[X_small_cols].to_numpy()
    y = d.LoadFactor.iloc[:].to_numpy()

    N, P_full = X_full.shape
    N, P_small = X_small.shape

    # train and test data 
    X = X_full if data_format == "onehot" else X_small
    X_train = X[~ix_val,:]
    y_train = y[~ix_val]
    X_val = X[ix_val,:]
    y_val = y[ix_val]

    return X, X_train, X_val, y, y_train, y_val



##################################### METRICS ########################################################

def relative_difference(y,y_hat):
    return (y-y_hat)/y
def accuracy(y,y_hat):
    return (1-np.abs(relative_difference(y,y_hat)))*100

def accuracy_score(y_true, y_pred, inv_trns = lambda x: x):
    return accuracy(y_true,inv_trns(y_pred)).mean()






##################################### MODELS ########################################################

def decision_tree(X, y, cv):
    trns = lambda x: x
    inv_trns = lambda x: x

    # setup tree
    dtree=DecisionTreeRegressor()
    score = make_scorer(accuracy_score, greater_is_better=True, inv_trns = inv_trns)
    param_grid = {
        'min_samples_leaf': range(1,200,5),
    }   

    cv_grid = GridSearchCV(estimator = dtree, param_grid = param_grid, cv = cv, verbose=2, n_jobs=-1,scoring=score)
    cv_grid.fit(X, trns(y))
    return cv_grid

def adaboost(X, y, cv):
    trns = lambda x: x
    inv_trns = lambda x: x
    score = make_scorer(accuracy_score, greater_is_better=True, inv_trns = inv_trns)

    # Try to experiment with max_samples, max_features, number of modles, and other models
    n_estimators = [50] #range(5,101, 5)
    max_depth = range(15,25)

    #We do an outer loop over max_depth here ourselves because we cannot include in the CV paramgrid.
    #Notice this is not a "proper" way to select the best max_depth but for the purpose of vizuallizing behaviour it should do
    # test_acc = np.zeros((len(n_estimators), len(max_depth)))
    cv_grid = {
        "max_depth" : max_depth,
        # "test_acc" : np.zeros((len(n_estimators), len(max_depth))),
        "boost_grid" : [None]*len(max_depth)
    }
    for j, i in enumerate(max_depth):
        
        # Create and fit an AdaBoosted decision tree
        boost = AdaBoostRegressor(DecisionTreeRegressor(max_depth = i), learning_rate= 1)

        params = {
            "n_estimators" : n_estimators,
            # "learning_rate" : np.arange(0.5, 2, .1)
        }

        boost_grid = GridSearchCV(boost, params, cv = cv, n_jobs = -1, verbose = 2, scoring = score)

        # Fit the grid search model
        boost_grid.fit(X, trns(y))


        # cv_grid["test_acc"][:,i-1] = boost_grid.cv_results_['mean_test_score']
        cv_grid["boost_grid"][j] = boost_grid
        cv_grid["max_depth"][j] = i

    return cv_grid



def fit_model(model_name, X, y, cv = None):
    if model_name == "decision_tree":
        return decision_tree(X, y, cv)
    elif model_name == "adaboost":
        return adaboost(X, y, cv)



if __name__ == "__main__":
    data_format = "summary_stats" # summary_stats or onehot
    X, X_train, X_val, y, y_train, y_val = get_data(data_format)
    
    cv = TimeSeriesSplit(n_splits=2) # either a TimeSeriesSplit(n_splits=2) or an int
    model_type = "adaboost"
    X_fit, y_fit = X_train, y_train

    global cv_grid
    cv_grid = fit_model(model_type, X_fit, y_fit, cv)
    

    save_name = f"models/model_{model_type}_data_{data_format}{'' if isinstance(cv, int) else '_tssplit'}.pkl"
    os.makedirs("models", exist_ok=True)
    with open(save_name, "wb") as file:
        pickle.dump(cv_grid, file)
    # joblib.dump(cv_grid, save_name)

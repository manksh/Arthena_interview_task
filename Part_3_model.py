"""
Author: Manksh Gupta

"""

# Imports

from utils import preprocess_all
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import time
import matplotlib.pyplot as plt

if __name__ == '__main__':

    print('Welcome to APPM - Art prices prediction model!')
    print('Author: Manksh Gupta, Columbia University')
    time.sleep(1)
    print('Lest Start!')
    time.sleep(1)
    print('Lets load the data!')
    time.sleep(1)
    # Read The existing data in
    picasso = pd.read_csv('artists/picasso.csv')
    lewitt = pd.read_csv('artists/lewitt.csv')
    warhol = pd.read_csv('artists/warhol.csv')
    print('Please enter full path for new artist training data:')
    # User Input for new artist training data
    new_train = input()
    print('Please enter full path for new artist test data:')
    # User Input for new artist testing data
    new_test = input()
    print('Awesome, Finishing up the loading!')
    time.sleep(1)
    train_new = pd.read_csv(new_train)
    test_new = pd.read_csv(new_test)
    dataset = pd.concat([picasso, lewitt, warhol, train_new, test_new])
    df = preprocess_all(dataset) # loaded from utils
    df = df.reset_index()
    df = df.drop(columns='index')

    # Doing some more preprocessing


    # Janky way to get test data after preprocessing together
    X_test = df.tail(test_new.shape[0])  
    X_test = X_test.drop(columns=['adjusted_hammer_price'])
    # Janky way to get train data after preprocessing together
    X_train = df.head(df.shape[0]-test_new.shape[0])
    X_train = X_train[X_train['adjusted_hammer_price'] > 0]
    y_train = X_train['adjusted_hammer_price']
    X_train = X_train.drop(columns=['adjusted_hammer_price'])

    print('Its all loaded - lets do some modeling!')
    time.sleep(1)
    print('Running a pooled Random Forest Model - Should be quick, hang tight!')

    regressor = RandomForestRegressor(random_state=0, n_estimators=150,
                                      n_jobs=-1, max_depth=5, min_samples_split=3)
    regressor.fit(X_train, y_train)
    print('Random Forest - Train R-squared', regressor.score(X_train, y_train))

    predicted_hammer_price = regressor.predict(X_test)
    predictions = pd.DataFrame(predicted_hammer_price)
    predictions.columns = ['predicted_hammer_price']
    predictions.to_csv('new_artist_predictions.csv')
    plt.hist(predictions)
    plt.savefig('prediction_histogram.png')

    print('Its all done, predictions are written as new_artist_predictions.csv - Enjoy!!')














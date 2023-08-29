import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

def retrain_Model_Tuned(new_dataset):
    model_df = pd.read_csv("Model/Model_History.csv")
    old_dataset = pd.read_csv(f"Model/dataset_v{len(model_df)}.csv")
    
    #Date Integration
    dataset = pd.concat([new_dataset, old_dataset])

    # export dataset
    dataset.to_csv(f'Model/dataset_v{len(model_df) + 1}.csv', index=False)

    #Pre-processing
    # drop unnecessary features
    drops = ['Number', 'title', 'artist']

    for drop in drops:
        dataset = dataset.drop(drop, axis=1)

    #Data Transformation
    # list of categorical columns
    categorical_cols = ['top genre']

    # initialize the OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)

    # fit and transform the selected categorical columns
    encoded_features = encoder.fit_transform(dataset[categorical_cols])
    
    # create a new DataFrame with the encoded features
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))

    # remove the original categorical columns from the original DataFrame
    dataset.drop(columns=categorical_cols, inplace=True)

    # Reset the index of both data frames
    dataset.reset_index(drop=True, inplace=True)
    encoded_df.reset_index(drop=True, inplace=True)

    # Concatenate the encoded DataFrame with the original DataFrame
    dataset = pd.concat([dataset, encoded_df], axis=1)

    #Data Selection
    # X for features and y for target
    X = dataset.drop('pop', axis=1)
    y = dataset['pop']

    # perform feature normalization using min-max scaling method
    # Create an instance of MinMaxScaler
    scaler = MinMaxScaler()

    # Fit the scaler to the data and transform the features
    X_scaled = scaler.fit_transform(X)

    #Data Splitting
    # split into testing and training datasets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    #Model Creation
    #Model Training
    # initialize the Decision Tree Regressor
    regressor = DecisionTreeRegressor()

    #Hyperparameter Tuning
    param_grid={"splitter":["best","random"],
            "max_depth" : [1,3,5,7,9,11,12],
            "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
            "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4,0.5],
            "max_features":["auto","log2","sqrt",None],
            "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90],
            "random_state": [5]
    }

    t1 = time.time()
    # perform grid search
    grid_search = GridSearchCV(regressor, param_grid, scoring='neg_mean_squared_error', cv=2, verbose=3)
    grid_search.fit(X_train, y_train)
    t2 = time.time()

    # display best parameters
    best_params = grid_search.best_params_

    # initialize tuned regressor
    tuned_regressor = grid_search.best_estimator_

    # fit the training datasets
    tuned_regressor.fit(X_train, y_train)

    #Model Evaluation
    # perform prediction using the test data
    y_predict = tuned_regressor.predict(X_test).astype(int)

    # model score in training dataset
    training_score = tuned_regressor.score(X_train, y_train)

    # model score in testing dataset
    testing_score = tuned_regressor.score(X_test, y_test)

    # compute for the Mean Square Error
    mse = mean_squared_error(y_test, y_predict)

    # compute for the Root Mean Square Error
    rmse = np.sqrt(mse)

    # display results
    print("Model Parameters:", best_params)
    print(f"Search Time: {(t2-t1)} secs")
    print(f"Training Score: {training_score * 100}%")
    print(f"Testing Score: {testing_score * 100}%")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")

    # save model summary
    model_summary = {'model version': [len(model_df) + 1],
                     'data': [len(dataset)],
                     'max_depth': [best_params['max_depth']],
                     'max_features': [best_params['max_features']],
                     'max_leaf_nodes': [best_params['max_leaf_nodes']],
                     'min_samples_leaf': [best_params['min_samples_leaf']],
                     'min_weight_fraction_leaf': [best_params['min_weight_fraction_leaf']],
                     'splitter': [best_params['splitter']],
                     'train score': [training_score],
                     'test score': [testing_score],
                     'mse': [mse],
                     'rmse': [rmse]
    }

    model_summary_df = pd.DataFrame(model_summary)

    model_df = pd.concat([model_df, model_summary_df], ignore_index=True)

    # export to csv
    model_df.to_csv('Model/Model_History.csv', index=False)

    #Model Export
    joblib.dump(tuned_regressor, "Decision_Tree_Regressor")
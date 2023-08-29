import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

def retrain_Model(new_dataset):
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
    regressor = DecisionTreeRegressor(criterion="squared_error", max_depth=5, min_samples_split=2, min_samples_leaf=7, random_state=5)

    
    # fit the training datasets
    regressor.fit(X_train, y_train)

    #Model Evaluation
    # perform prediction using the test data
    y_predict = regressor.predict(X_test).astype(int)

    # model score in training dataset
    training_score = regressor.score(X_train, y_train)

    # model score in testing dataset
    testing_score = regressor.score(X_test, y_test)

    # compute for the Mean Square Error
    mse = mean_squared_error(y_test, y_predict)

    # compute for the Root Mean Square Error
    rmse = np.sqrt(mse)

    # display results
    print(f"Training Score: {training_score * 100}%")
    print(f"Testing Score: {testing_score * 100}%")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")

    # save model summary
    model_summary = {'model version': [len(model_df) + 1],
                     'data': [len(dataset)],
                     'max_depth': [0],
                     'max_features': [0],
                     'max_leaf_nodes': [0],
                     'min_samples_leaf': [0],
                     'min_weight_fraction_leaf': [0],
                     'splitter': [0],
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
    joblib.dump(regressor, "Decision_Tree_Regressor")
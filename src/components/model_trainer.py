import os, sys
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.ensemble import VotingRegressor

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models,print_evaluated_results,model_metrics,load_object,save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer = ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('Spliting dependent and independent variables from train and test data')
            xtrain, ytrain, xtest, ytest = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info('Spliting done')

            logging.info('Model Selection Started')

            models = {
                'Linear Regression': LinearRegression(),
                # 'SVR': SVR(),
                # 'Lasso': Lasso(),
                # 'Ridge': Ridge(),
                'K-Neighbors Regressor': KNeighborsRegressor(),
                'Random Forest Regressor': RandomForestRegressor(),
                'AdaBoost Regressor': AdaBoostRegressor(),
                'Gradient Boosting Regressor': GradientBoostingRegressor(),
                'Decision Tree': DecisionTreeRegressor(), 
            }

            logging.info('Diverted to evaluate_models in utils')
            model_report=evaluate_models(xtrain,ytrain,xtest,ytest,models)

            print(model_report)
            print('\n====================================================\n')
            logging.info(f"Model Report : {model_report}")

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                logging.info('No Best Model has r2 score less than 60%')
                raise CustomException('No Best Model found')
            
            print(f"Best Model Found, Model Name : {best_model_name}, R2 Score : {best_model_score}")

            print("\n====================================================\n")
            logging.info(f"Best Model Found, Model name : {best_model_name}, R2 Score : {best_model_score}")
            logging.info('Hypertuning started for Random forest Regressor')

            rf = RandomForestRegressor()
            params = {
                'n_estimators': [10,20,50],
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }

            rf_random = RandomizedSearchCV(rf, params, cv=5, scoring='r2',n_jobs=-1)
            rf_random.fit(xtrain,ytrain)

            print(f"Best Random Forest Parameters : {rf_random.best_params_}")
            print(f"Best Random Forest Score : {rf_random.best_score_}")
            print("\n=======================================================\n")

            best_rf = rf_random.best_estimator_

            logging.info('Hyperparameter tuning completed for Random Forest')

            logging.info('Hyperparameter tuning for KN Regregessor started')

            knn = KNeighborsRegressor()

            k_range = list(range(2,31))
            param_grid = dict(n_neighbors=k_range)

            grid = GridSearchCV(knn,param_grid,cv=5,scoring='r2',n_jobs=-1)
            grid.fit(xtrain,ytrain)

            print(f"Best K-Neighbors Parameters : {grid.best_params_}")
            print(f"Best K-Neighbors Score : {grid.best_score_}")
            print("\n=======================================================\n")

            best_knn = grid.best_estimator_

            logging.info("Hyperparameter tuning completed for K-Neighbors")

            logging.info("Hyperparameter tuning for Decision Tree Started")

            dtr = DecisionTreeRegressor()

            param = {
                "splitter":["best","random"],
                "max_depth" : [1,3,5,7],
                "min_samples_leaf":[1,2,3,4,5,6,]
            }

            grid_dec = GridSearchCV(dtr,param,cv=5,scoring='r2',n_jobs=-1)
            grid_dec.fit(xtrain,ytrain)

            print(f"Best Decision Tree Parameters : {grid_dec.best_params_}")
            print(f"Best Decision Tree Score : {grid_dec.best_score_}")
            print("\n=======================================================\n")

            best_dtr = grid_dec.best_estimator_

            logging.info('Hyperparameter tuning complete for Decision Tree')

            logging.info('VotingRegressor model training started')

            er = VotingRegressor([('rf',best_rf),('knn',best_knn),('dtr',best_dtr)],weights=[3,2,1])
            er.fit(xtrain,ytrain)
            print('Final Model Evaluation : \n')
            print_evaluated_results(xtrain,ytrain,xtest,ytest,er)
            logging.info('Voting Regressor training completed')

            save_object(
                file_path=self.model_trainer.trained_model_file_path,
                obj=er
            )
            logging.info('Model Pickle file savel')

            ytest_pred = er.predict(xtest)

            mae, rmse, r2 = model_metrics(ytest,ytest_pred)
            logging.info(f"Test MAE : {mae}")
            logging.info(f"Test RMSE : {rmse}")
            logging.info(f"Test R2 Score: {r2}")
            logging.info("Final Model Training Completed")

            return mae, rmse, r2


        except Exception as e:
            logging.info('Exception occured in Initiate_Model_Training')
            raise CustomException(e,sys)
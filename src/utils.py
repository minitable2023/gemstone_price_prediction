import os, sys
import dill
import numpy as np
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


from src.exception import CustomException
from src.logger import logging


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(xtrain,ytrain,xtest,ytest,models):
    logging.info('Evaluate_model Started')
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(xtrain,ytrain)

            ytrain_pred = model.predict(xtrain)
            ytest_pred = model.predict(xtest)

            train_model_score = r2_score(ytrain,ytrain_pred)
            test_model_score = r2_score(ytest,ytest_pred)

            report[list(models.keys())[i]] = test_model_score

            logging.info('Score recieved')

        return report
        

    except Exception as e:
        logging.info('Exception occured in evaluate_models')
        raise CustomException(e,sys)
    
def model_metrics(true,pred):
    try:
        mae = mean_absolute_error(true,pred)
        mse = mean_squared_error(true,pred)
        rmse = np.sqrt(mse)
        r2_square = r2_score(true,pred)
        return mae, rmse, r2_square

    except Exception as e:
        logging.info('Exception occured while evaluating metrics')
        raise CustomException(e,sys)
    
def print_evaluated_results(xtrain,ytrain,xtest,ytest,model):
    try:
        ytrain_pred = model.predict(xtrain)
        ytest_pred = model.predict(xtest)

        model_train_mae, model_train_rmse, model_train_r2 = model_metrics(ytrain,ytrain_pred)
        model_test_mae, model_test_rmse, model_test_r2 = model_metrics(ytest,ytest_pred)

        print('Model Performance fot Training Set')
        print('- Root Mean Square Error : {:.4f}'.format(model_train_rmse))
        print('- Mean Absolute Error : {:.4f}'.format(model_train_mae))
        print("- R2 Score : {:.4f}".format(model_train_r2))

        print('-'*35)

        print('Model Performance fot Test Set')
        print('- Root Mean Square Error : {:.4f}'.format(model_test_rmse))
        print('- Mean Absolute Error : {:.4f}'.format(model_test_mae))
        print("- R2 Score : {:.4f}".format(model_test_r2))


    except Exception as e:
        logging.info('Exception occured in printing evaluated results')
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        logging.info('Exception occured in load_object function in utils')
        raise CustomException(e,sys)
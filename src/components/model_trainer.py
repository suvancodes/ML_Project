from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import r2_score
import os, sys
from src.utlit import evalute_model, save_obj
from src.exception import CustomException
from src.logger import logging


@dataclass
class modeltraningconfig:
    trained_model_file_path = os.path.join('artifact', 'model.pkl')


class ModelTraning:
    def __init__(self):
        self.model_trainer_config = modeltraningconfig()

    def start_model_traing(self, train_array, test_array):
        try:
            logging.info("Splitting train and test data")

            # ✅ Split features and target correctly
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'RandomForestRegressor': RandomForestRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'LinearRegression': LinearRegression(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'GaussianNB': GaussianNB(),
                'CatBoostRegressor': CatBoostRegressor(verbose=0)
            }

            model_report: dict = evalute_model(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models
            )

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found")

            logging.info(f"Best model found: {best_model_name} with score {best_model_score}")

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            pred = best_model.predict(X_test)
            return r2_score(y_test, pred)

        except Exception as e:
            raise CustomException(e, sys)

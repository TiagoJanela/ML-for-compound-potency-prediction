import itertools
from typing import *
import numpy as np
import pandas as pd

# Tensorflow/Keras
import keras
import tensorflow as tf
from tensorflow import keras
from keras import backend as K, Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Lambda, Input

# Sklearn
from sklearn import neighbors, metrics
from sklearn.metrics import mean_absolute_error
from sklearn.dummy import DummyRegressor
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from ml_utils import tanimoto_from_sparse
import warnings

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None


class Dataset:
    def __init__(self, features: np.array, labels: np.array):
        self.features = features
        self.labels = labels
        self._add_instances = set()

    def add_instance(self, name, values: np.array):
        self._add_instances.add(name)
        self.__dict__[name] = values

    @property
    def columns(self) -> dict:
        data_dict = {k: v for k, v in self.__dict__.items()}
        data_dict['features'] = self.features
        data_dict['labels'] = self.labels
        return data_dict

    def __len__(self):
        return self.labels.shape[0]

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return {col: values[idx] for col, values in self.columns.items()}

        subset = Dataset(self.features[idx], self.labels[idx])
        for addt_instance in self._add_instances:
            subset.add_instance(addt_instance, self.__dict__[addt_instance][idx])

        return subset


class MLModel:
    def __init__(self, data, ml_algorithm, reg_class="regression",
                 parameters='grid', cv_fold=10, random_seed=2002):

        self.data = data
        self.ml_algorithm = ml_algorithm
        self.reg_class = reg_class
        self.cv_fold = cv_fold
        self.seed = random_seed
        self.parameters = parameters
        self.h_parameters = self.hyperparameters()
        self.model, self.cv_results = self.cross_validation()
        self.best_params = self.optimal_parameters()
        self.model = self.final_model()

    def hyperparameters(self):
        if self.parameters == "grid":
            if self.reg_class == "regression":
                if self.ml_algorithm == "MR":
                    return {'strategy': ['median']
                            }
                elif self.ml_algorithm == "SVR":
                    return {'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, 100, 10000],
                            'kernel': [tanimoto_from_sparse],
                            }
                elif self.ml_algorithm == "RFR":
                    return {'n_estimators': [25, 100, 200],
                            'max_features': ['auto'],
                            'min_samples_split': [2, 3, 5],
                            'min_samples_leaf': [1, 2, 5],
                            }
                elif self.ml_algorithm == "kNN":
                    return {"n_neighbors": [1, 3, 5]
                            }

            if self.reg_class == "classification":
                if self.ml_algorithm == "SVM":
                    return {'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10],
                            'kernel': [tanimoto_from_sparse],
                            }
                elif self.ml_algorithm == "RFC":
                    return {'n_estimators': [25, 100, 200],
                            'max_features': ['auto'],
                            'min_samples_split': [2, 3, 5],
                            'min_samples_leaf': [1, 2, 5],
                            }


    def cross_validation(self):

        if self.reg_class == "regression":
            opt_metric = "neg_mean_absolute_error"
            if self.ml_algorithm == "MR":
                model = DummyRegressor()
            elif self.ml_algorithm == "SVR":
                model = SVR()
            elif self.ml_algorithm == "RFR":
                model = RandomForestRegressor(random_state=self.seed)
            elif self.ml_algorithm == "kNN":
                model = neighbors.KNeighborsRegressor()

        elif self.reg_class == "classification":
            opt_metric = "balanced_accuracy"
            if self.ml_algorithm == "SVM":
                model = SVC()
            elif self.ml_algorithm == "RFC":
                model = RandomForestClassifier()

        cv_results = GridSearchCV(model,
                                  param_grid=self.h_parameters,
                                  cv=self.cv_fold,
                                  scoring=opt_metric,
                                  n_jobs=-1)
        cv_results.fit(self.data.features, self.data.labels)

        return model, cv_results

    def optimal_parameters(self):
        best_params = self.cv_results.cv_results_['params'][self.cv_results.best_index_]
        return best_params

    def final_model(self):
        model = self.model.set_params(**self.best_params)
        return model.fit(self.data.features, self.data.labels)


class DNN:
    def __init__(self, data, ml_algorithm, n_features, seed, reg_class="regression", parameters='grid'):

        self.data = data
        self.ml_algorithm = ml_algorithm
        self.n_features = n_features
        self.reg_class = reg_class
        self.parameters = parameters
        self.seed = seed
        self.h_parameters = self.dnn_hyperparameters()
        self.model = self.dnn_model()
        self.cv_results = self.dnn_cross_validation()
        self.best_params = self.dnn_select_best_parameters()
        self.model = self.final_model()
        self.final_model = self.fit_model()

    def dnn_hyperparameters(self):
        if self.parameters == "grid":
            return {
                "layers": [(100, 100), (250, 250), (250, 500), (500, 250), (500, 250, 100), (100, 250, 500)],
                "dropout": [0.0, 0.25, 0.5],
                "activation": ['tanh'],
                "learning_rate": [0.1, 0.01, 0.001],
                "n_epochs": [200]
            }

    def dnn_model(self, layers=None, dropout_rate: float = 0, activation_f: str = "tanh",
                  learning_rate: float = 0.01, n_features: int = None, seed: int = None, ):

        #set seed
        tf.keras.utils.set_random_seed(self.seed)
        tf.config.experimental.enable_op_determinism()

        if layers is None:
            layers = (100, 100)

        inputs = keras.Input(shape=(self.n_features,), name="input_layer")

        for i, net_nodes in enumerate(layers, 1):
            if i == 1:
                layer = Dense(int(net_nodes), activation=activation_f, name=f'Dense_{i}')(inputs)
                layer = Dropout(dropout_rate)(layer)
            elif i == len(layers):
                if self.reg_class == "regression":
                    layer_reg = Dense(1, activation="linear")(layer)
                else:
                    layer_class = Dense(2, activation="sigmoid")(layer)
            else:
                layer = Dense(int(net_nodes), activation=activation_f, name=f'Dense_{i}')(layer)
                layer = Dropout(dropout_rate)(layer)

        if self.reg_class == "regression":
            model = Model(inputs, layer_reg, name='DNN_model')
            loss = 'mean_absolute_error'
            opt_metric = ["mean_absolute_error"]
        else:
            model = Model(inputs, layer_class, name='DNN_model')
            loss = 'sparse_categorical_crossentropy'
            opt_metric = ["accuracy"]

        model.compile(loss=loss,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      metrics=opt_metric)

        return model

    def dnn_cross_validation(self) -> Dict:

        """
         Grid Search for parameter optimization

        """
        # initialize hyper-parameters
        hyperparameters = self.h_parameters

        # Build grid for all parameters
        parameter_grid = itertools.product(*hyperparameters.values())

        grid_search_results = dict()

        for i, grid_comb in enumerate(parameter_grid):
            # Start Grid-Search
            nn_layers, dropout_rate, activation, learning_rate, train_epochs = grid_comb

            # Build the model for each grid setup
            model = self.dnn_model(layers=nn_layers,
                                   dropout_rate=dropout_rate,
                                   activation_f=activation,
                                   learning_rate=learning_rate,
                                   n_features=self.n_features,
                                   )

            # EarlyStopping
            earlystopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0)

            # Model training and validation
            history = model.fit(self.data.features.toarray(), self.data.labels,
                                batch_size=32,
                                epochs=train_epochs,
                                validation_split=0.2,
                                callbacks=[earlystopping],
                                verbose=0)

            # Retrieve the trained parameters from the model
            grid_search_results[grid_comb] = history

        return grid_search_results

    def dnn_select_best_parameters(self):

        """
        Grid Search selection of best parameters

        """
        grid_search_results = []

        for param_comb, fitted_model in self.cv_results.items():
            # Model Training history
            history_data = pd.DataFrame(fitted_model.history)

            # Save stopping epoch
            history_data = history_data.reset_index().rename(columns={"index": "on_epoch"})
            history_data["on_epoch"] = history_data["on_epoch"].apply(lambda x: x + 1)

            # Select loss with minimum validation loss
            best_per_model = history_data.loc[history_data["val_loss"].idxmin(skipna=True)].rename(param_comb)
            grid_search_results.append(best_per_model)

        # Concatenate all models training results
        grid_search_results = pd.concat(grid_search_results, axis=1).T

        # select optimal hyperparameter settings
        optimal_stats = grid_search_results.loc[grid_search_results["val_loss"].idxmin(skipna=True)]
        opt_setting = {k: v for k, v in zip(self.h_parameters.keys(), optimal_stats.name)}
        opt_setting.update(optimal_stats.to_dict())

        return opt_setting

    def final_model(self):

        dnn = self.dnn_model(self.best_params["layers"],
                             self.best_params["dropout"],
                             self.best_params["activation"],
                             self.best_params["learning_rate"],
                             n_features=self.n_features
                             )

        return dnn

    def fit_model(self):

        # define early stopping object
        earlystopping = EarlyStopping(monitor='loss', patience=10, verbose=0)

        # train the model on the entire training set
        best_model = self.model.fit(self.data.features.toarray(), self.data.labels,
                                    batch_size=32,
                                    epochs=self.best_params['n_epochs'],
                                    validation_split=None,
                                    callbacks=[earlystopping],
                                    verbose=0)

        return best_model


class Model_Evaluation:
    def __init__(self, model, data, data_transformer=None, model_id=None, reg_class="regression"):
        self.reg_class = reg_class
        self.model_id = model_id
        self.data_transformer = data_transformer
        self.model = model
        self.data = data
        self.labels, self.y_pred, self.predictions = self.model_predict(data)
        self.pred_performance = self.prediction_performance(data)

    def model_predict(self, data):

        if self.model_id == 'GCN':
            y_prediction = self.data_transformer.untransform(self.model.predict(data).flatten())
            labels = self.data_transformer.untransform(data.y)

        elif self.model.ml_algorithm == 'DNN':
            y_prediction = self.model.model.predict(data.features.toarray(), verbose=0).flatten()
            labels = data.labels

        else:
            y_prediction = self.model.model.predict(data.features)
            labels = data.labels

        predictions = pd.DataFrame(list(zip(labels, y_prediction)), columns=["true", "predicted"])

        if self.model_id == 'GCN':
            predictions['Target ID'] = data.ids
            predictions['algorithm'] = self.model_id
        else:
            predictions['Target ID'] = data.target[0]
            predictions['algorithm'] = self.model.ml_algorithm

        return labels, y_prediction, predictions

    def prediction_performance(self, data, y_score=None, nantozero=False) -> pd.DataFrame:

        if self.reg_class == 'classification':
            fill = 0 if nantozero else np.nan
            if sum(self.y_pred) == 0:
                mcc = fill
                precision = fill
            else:
                mcc = metrics.matthews_corrcoef(data.labels, self.y_pred)
                precision = metrics.precision_score(data.labels, self.y_pred)

            result_list = [{"MCC": mcc,
                            "F1": metrics.f1_score(data.labels, self.y_pred),
                            "BA": metrics.balanced_accuracy_score(data.labels, self.y_pred),
                            "Precision": precision,
                            "Recall": metrics.recall_score(data.labels, self.y_pred),
                            "Average Precision": metrics.average_precision_score(data.labels, self.y_pred),
                            "data_set_size": data.labels.shape[0],
                            "true_pos": len([x for x in data.labels if x == 1]),
                            "true_neg": len([x for x in data.labels if x == 0]),
                            "predicted_pos": len([x for x in self.y_pred if x == 1]),
                            "predicted_neg": len([x for x in self.y_pred if x == 0])},
                           ]

            if y_score is not None:
                result_list.append({"AUC": metrics.roc_auc_score(data.labels, self.y_pred)})
            else:
                result_list.append({"AUC": np.nan})

            results = pd.DataFrame(result_list)
            results = results[
                ["Target ID", "Algorithm", "MCC", "F1", "BA", "Precision", "Recall", "Average Precision"]]
            results["Target ID"] = results["Target ID"].map(lambda x: x.lstrip("CHEMBL").rstrip(""))
            results.set_index(["Target ID", "Algorithm"], inplace=True)
            results.columns = pd.MultiIndex.from_product([["Value"],
                                                          ["MCC", "F1", "BA", "Precision", "Recall",
                                                           "Average Precision"]], names=["Value", "Metric"])
            results = results.stack().reset_index().set_index("Target ID")

            return results

        elif self.reg_class == "regression":

            labels = self.labels
            pred = self.y_pred

            fill = 0 if nantozero else np.nan
            if len(pred) == 0:
                mae = fill
                mse = fill
                rmse = fill
                r2 = fill
            else:
                mae = mean_absolute_error(labels, pred)
                mse = metrics.mean_squared_error(labels, pred)
                rmse = metrics.mean_squared_error(labels, pred, squared=False)
                r2 = metrics.r2_score(labels, pred)

            if self.model_id == 'GCN':
                target = list(self.data.ids)[0]
                model_name = self.model_id
            else:
                target = data.target[0]
                model_name = self.model.ml_algorithm

            result_list = [{"MAE": mae,
                            "MSE": mse,
                            "RMSE": rmse,
                            "R2": r2,
                            "data_set_size": len(labels),
                            "Target ID": target,
                            "Algorithm": model_name}
                           ]

            # Prepare result dataset
            results = pd.DataFrame(result_list)
            results = results[["Target ID", "Algorithm", "MAE", "MSE", "RMSE", "R2"]]
            results["Target ID"] = results["Target ID"].map(lambda x: x.lstrip("CHEMBL").rstrip(""))
            results.set_index(["Target ID", "Algorithm"], inplace=True)
            results.columns = pd.MultiIndex.from_product([["Value"], ["MAE", "MSE", "RMSE", "R2"]],
                                                         names=["Value", "Metric"])
            results = results.stack().reset_index().set_index("Target ID")

            return results


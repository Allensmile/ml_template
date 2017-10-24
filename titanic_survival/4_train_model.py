#!/usr/bin/env python

import numpy as np
import scipy as sp
import pandas as pd
import pprint
import math
import pickle
import sklearn
import xgboost
import lightgbm

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import TransformerMixin


def main():
  id_feature_name = "PassengerId"
  label_feature_name = "Survived"

  train_df, test_df = load_train_test_files()
  trainable_df = drop_id_feature(train_df, id_feature_name)
  preditable_df = drop_id_feature(test_df, id_feature_name)
  model = train_classification_model(trainable_df, label_feature_name)
  predictions = predict_test_dataset(model, preditable_df)
  submission_df = pd.DataFrame({
      "PassengerId": test_df[id_feature_name],
      "Survived": predictions
  })
  export_submission_file(submission_df)
  """
  tune_regression_model(trainable_df, label_feature_name)
  """


def titanic_main():
  #train_csv_file_path = "./train_fe1.csv"
  train_csv_file_path = "./train_fe1.csv"
  test_csv_file_path = "./test_fe1.csv"

  dataset = pd.read_csv(train_csv_file_path)
  #test_dataset = pd.read_csv(test_csv_file_path)

  view_sample_dataset(dataset)
  drop_useless_features(dataset)
  train_model(dataset)


def load_train_test_files():
  print("\n[Debug] Load the train and test files: ")
  train_csv_file_path = "./generated/train_feature.csv"
  test_csv_file_path = "./generated/test_feature.csv"
  train_df = pd.read_csv(train_csv_file_path)
  test_df = pd.read_csv(test_csv_file_path)
  print("Read train file: {} and test file: {}".format(train_csv_file_path,
                                                       test_csv_file_path))

  return train_df, test_df


def view_sample_dataset(dataset):
  print("\n[Debug] Print the sample of the dataset: ")
  dataset_sample = dataset.head(1)
  print(dataset_sample)


def drop_useless_features(dataset):
  print("\n[Debug] Drop the useless features of the dataset: ")
  useless_feature_name_array = ["Unnamed: 0"]
  print("Drop the features: {}".format(useless_feature_name_array))
  dataset.drop(useless_feature_name_array, axis=1, inplace=True)


def drop_id_feature(df, id_feature_name):
  """
  Return: the dataframe without the id feature.
  """
  print("\n[Debug] Drop the id feature: ")
  print("Drop id feature: {}".format(id_feature_name))
  return df.drop(id_feature_name, axis=1, inplace=False)


def train_regression_model(df, label_feature_name):
  print("\n[Debug] Train the regression model: ")

  train_X, train_Y, test_X, test_Y = _split_train_test_feature_label(
      df, label_feature_name)
  model1 = xgboost.XGBRegressor()
  model2 = sklearn.ensemble.GradientBoostingRegressor()  # 0.13
  model = AveragingModels([model1, model2])
  model = _train_with_model(model, train_X, train_Y)
  _compute_target_score_for_test(model, test_X, test_Y)

  return model


def train_classification_model(df, label_feature_name):
  print("\n[Debug] Train the classification model: ")

  train_X, train_Y, test_X, test_Y = _split_train_test_feature_label(
      df, label_feature_name)
  model = xgboost.XGBClassifier()
  model = _train_with_model(model, train_X, train_Y)
  #_compute_target_score_for_test(model, test_X, test_Y)

  return model


def _split_train_test_feature_label(df, label_feature_name):
  print("\n[Debug] Split the train/test feature/label dataset: ")

  #train, test = train_test_split(df, test_size=0.3, random_state=0, stratify=df[label_feature_name])
  train, test = train_test_split(df, test_size=0.2, random_state=0)

  train_X = train.drop(label_feature_name, axis=1, inplace=False)
  train_Y = train[label_feature_name]
  test_X = test.drop(label_feature_name, axis=1, inplace=False)
  test_Y = test[label_feature_name]

  return train_X, train_Y, test_X, test_Y


def _train_with_model(model, train_X, train_Y):
  print("\n[Debug] Train wth the model: ")

  model.fit(train_X, train_Y)
  print("Success to train the model: {}".format(model))

  return model


def _compute_target_score_for_test(model, test_X, test_Y):
  print("\n[Debug] Compute the target score for test dataset: ")

  predict_Y = model.predict(test_X)

  # TODO: Prevent from getting negative number
  predict_Y = predict_Y.clip(min=1)

  root_mean_square_error = mean_squared_error(
      np.log(test_Y), np.log(predict_Y))
  score = math.sqrt(root_mean_square_error)
  print("The target score is: {}".format(score))

  return score


def tune_regression_model(df, label_feature_name):
  print("\n[Debug] Tune the regression model: ")
  """
  train_X, train_Y, test_X, test_Y = _split_train_test_feature_label(df, label_feature_name)
  pickle.dump(train_X, open("generated/train_X.pickle", "wb"))
  pickle.dump(train_Y, open("generated/train_Y.pickle", "wb"))
  pickle.dump(test_X, open("generated/test_X.pickle", "wb"))
  pickle.dump(test_Y, open("generated/test_Y.pickle", "wb"))
  """

  train_X = pickle.load(open("generated/train_X.pickle", "rb"))
  train_Y = pickle.load(open("generated/train_Y.pickle", "rb"))
  test_X = pickle.load(open("generated/test_X.pickle", "rb"))
  test_Y = pickle.load(open("generated/test_Y.pickle", "rb"))

  # Refer to http://scikit-learn.org/stable/modules/linear_model.html
  model = sklearn.linear_model.LinearRegression()  # 1.02
  model = sklearn.linear_model.Ridge(alpha=0.5)  # 0.68
  model = sklearn.linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])  # 0.17
  model = sklearn.linear_model.Lasso(alpha=0.1)  # 0.78
  model = sklearn.linear_model.LassoLars(alpha=0.1)  # 0.77
  model = sklearn.linear_model.ElasticNet(alpha=0.1)  # 0.17
  model = sklearn.linear_model.BayesianRidge(n_iter=300)  # 0.17
  model = sklearn.linear_model.ARDRegression()  # 0.16
  model = sklearn.linear_model.HuberRegressor()  # 0.22
  model = sklearn.linear_model.Perceptron()  # 0.41
  model1 = xgboost.XGBRegressor()  # 0.13
  model2 = xgboost.XGBRegressor(
      colsample_bytree=0.4603,
      gamma=0.0468,
      learning_rate=0.05,
      max_depth=3,
      min_child_weight=1.7817,
      n_estimators=2200,
      reg_alpha=0.4640,
      reg_lambda=0.8571,
      subsample=0.5213,
      silent=1,
      random_state=7,
      nthread=-1)
  model3 = sklearn.ensemble.GradientBoostingRegressor()  # 0.13
  model4 = sklearn.ensemble.RandomForestRegressor()  # 0.15
  model5 = lightgbm.LGBMRegressor(
      objective='regression',
      num_leaves=5,  # 0.21
      learning_rate=0.05,
      n_estimators=720,
      max_bin=55,
      bagging_fraction=0.8,
      bagging_freq=5,
      feature_fraction=0.2319,
      feature_fraction_seed=9,
      bagging_seed=9,
      min_data_in_leaf=6,
      min_sum_hessian_in_leaf=11)

  model = AveragingModels([model1, model2, model3])

  model = _train_with_model(model, train_X, train_Y)
  #_compute_target_score_for_test(model, test_X, test_Y)

  return model


def predict_test_dataset(model, df):
  print("\n[Debug] Predict with the model: ")

  predictions = model.predict(df)
  print("Success to predict with the model")
  print(predictions)

  return predictions


def export_submission_file(submission_df):
  print("\n[Debug] Export to submission file: ")
  my_submission_file_path = "./generated/my_submission.csv"
  print("Export the submission to file: {}".format(my_submission_file_path))
  submission_df.to_csv(my_submission_file_path, index=False)


def train_regression_model_old(df, test_df):
  #model = RandomForestRegressor()
  model = GradientBoostingRegressor()

  #selected_feature_name_array = ['GarageArea','YearRemodAdd','YearBuilt','1stFlrSF', '2ndFlrSF', 'TotalBsmtSF', 'BsmtUnfSF']
  selected_feature_name_array = [
      'YearRemodAdd', 'YearBuilt', '1stFlrSF', '2ndFlrSF'
  ]
  X_train = df[selected_feature_name_array]
  y_train = df["SalePrice"]

  model.fit(X_train, y_train)
  print("Success to train the model")

  y_pred = model.predict(X_train)
  root_mean_square_error = mean_squared_error(np.log(y_train), np.log(y_pred))
  score = math.sqrt(root_mean_square_error)
  print("The target score is: {}".format(score))

  X_test = test_df[selected_feature_name_array]
  predictions = model.predict(X_test)
  print(predictions)

  submission = pd.DataFrame({"Id": test_df["Id"], "SalePrice": predictions})
  my_submission_file_path = "./generated/my_submission.csv"
  print("Export the submission to file: {}".format(my_submission_file_path))
  submission.to_csv(my_submission_file_path, index=False)


def train_model(dataset):
  print("\n[Debug] Train the model of the dataset: ")

  train, test = train_test_split(
      dataset, test_size=0.3, random_state=0, stratify=dataset['Survived'])
  train_X = train[train.columns[1:]]
  train_Y = train[train.columns[:1]]
  test_X = test[test.columns[1:]]
  test_Y = test[test.columns[:1]]
  all_X = dataset[dataset.columns[1:]]
  all_Y = dataset['Survived']

  print(train_X.head(1))

  model = svm.SVC(kernel='rbf', C=1, gamma=0.1)
  model.fit(train_X, train_Y)
  prediction1 = model.predict(test_X)
  print('Accuracy for rbf SVM is ', metrics.accuracy_score(
      prediction1, test_Y))

  model = svm.SVC(kernel='linear', C=0.1, gamma=0.1)
  model.fit(train_X, train_Y)
  prediction2 = model.predict(test_X)
  print('Accuracy for linear SVM is', metrics.accuracy_score(
      prediction2, test_Y))

  model = LogisticRegression()
  model.fit(train_X, train_Y)
  prediction3 = model.predict(test_X)
  print('The accuracy of the Logistic Regression is', metrics.accuracy_score(
      prediction3, test_Y))

  model = DecisionTreeClassifier()
  model.fit(train_X, train_Y)
  prediction4 = model.predict(test_X)
  print('The accuracy of the Decision Tree is', metrics.accuracy_score(
      prediction4, test_Y))

  model = KNeighborsClassifier()
  model.fit(train_X, train_Y)
  prediction5 = model.predict(test_X)
  print('The accuracy of the KNN is', metrics.accuracy_score(
      prediction5, test_Y))

  model = GaussianNB()
  model.fit(train_X, train_Y)
  prediction6 = model.predict(test_X)
  print('The accuracy of the NaiveBayes is', metrics.accuracy_score(
      prediction6, test_Y))

  model = RandomForestClassifier(n_estimators=100)
  model.fit(train_X, train_Y)
  prediction7 = model.predict(test_X)
  print('The accuracy of the Random Forests is', metrics.accuracy_score(
      prediction7, test_Y))
  """
  from sklearn.model_selection import GridSearchCV
  C=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
  gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
  kernel=['rbf','linear']
  hyper={'kernel':kernel,'C':C,'gamma':gamma}
  gd=GridSearchCV(estimator=svm.SVC(),param_grid=hyper,verbose=True)
  gd.fit(X,Y)
  print(gd.best_score_)
  print(gd.best_estimator_)


  n_estimators=range(100,1000,100)
  hyper={'n_estimators':n_estimators}
  gd=GridSearchCV(estimator=RandomForestClassifier(random_state=0),param_grid=hyper,verbose=True)
  gd.fit(X,Y)
  print(gd.best_score_)
  print(gd.best_estimator_)
  """

  model = svm.SVC(kernel='rbf', C=1, gamma=0.1)
  model.fit(train_X, train_Y)
  prediction1 = model.predict(test_X)
  print('Accuracy for rbf SVM is ', metrics.accuracy_score(
      prediction1, test_Y))

  test_csv_file_path = "./test_fe1.csv"
  test_dataset = pd.read_csv(test_csv_file_path)
  new_test_data = test_dataset.drop(["PassengerId"], axis=1, inplace=False)
  new_test_data = new_test_data.drop(["Unnamed: 0"], axis=1, inplace=False)
  #new_test_data = new_test_data.drop(["Survived"], axis=1, inplace=False)
  #import ipdb;ipdb.set_trace()

  predictions = model.predict(new_test_data)
  submission = pd.DataFrame({
      "PassengerId": test_dataset["PassengerId"],
      "Survived": predictions
  })
  submission.to_csv("submission_fe1.csv", index=False)


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
  def __init__(self, models):
    self.models = models

  def fit(self, X, y):
    #self._models = [clone(x) for x in self.models]
    self._models = self.models
    for model in self._models:
      model.fit(X, y)
    return self

  def predict(self, X):
    predictions = np.column_stack([model.predict(X) for model in self._models])
    return np.mean(predictions, axis=1)


if __name__ == "__main__":
  #smoke_main()
  main()

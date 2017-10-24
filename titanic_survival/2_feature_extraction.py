#!/usr/bin/env python

import numpy as np
import scipy as sp
import pandas as pd
import pprint


def smoke_main():
  selected_train_features = [
      "SalePrice", 'GarageArea', 'YearRemodAdd', 'YearBuilt', '1stFlrSF',
      '2ndFlrSF', 'TotalBsmtSF', 'BsmtUnfSF'
  ]
  selected_test_features = [
      "Id", 'GarageArea', 'YearRemodAdd', 'YearBuilt', '1stFlrSF', '2ndFlrSF',
      'TotalBsmtSF', 'BsmtUnfSF'
  ]

  train_df, test_df = load_train_test_dataframes()
  train_df = select_features(train_df, selected_train_features)
  test_df = select_features(test_df, selected_test_features)
  export_train_test_dataframes(train_df, test_df)


def main():
  useless_features = ["Name", "Ticket"]
  missing_values_features = ["Cabin"]
  fill_most_features = ["Embarked"]
  fill_zero_features = []
  fill_mean_features = ["Age"]
  string_features = []
  converted_features = ["Age", "Fare", "Sex", "Embarked"]
  valid_string_feature_set = set(string_features) - set(missing_values_features)
  valid_string_features = [feature for feature in valid_string_feature_set]
  log_numberic_features = []

  train_df, test_df = load_train_test_dataframes()

  df = train_df
  drop_useless_features(df, useless_features)
  drop_missing_values_features(df, missing_values_features)
  fill_nan_value(df, fill_most_features, fill_mean_features,
                 fill_zero_features)
  feature_name_unique_value_map = onehot_encode_string_features_for_train(
      df, valid_string_features)
  drop_string_features(df, valid_string_features)
  compute_log_numberic_features(df, log_numberic_features)
  #generate_sum_features(df)
  #generate_has_features(df)
  drop_converted_features(df, converted_features)
  _preview_examples(df)

  df = test_df
  drop_useless_features(df, useless_features)
  drop_missing_values_features(df, missing_values_features)
  fill_nan_value(df, fill_most_features, fill_mean_features,
                 fill_zero_features)
  onehot_encode_string_features_for_test(df, valid_string_features,
                                         feature_name_unique_value_map)
  drop_string_features(df, valid_string_features)
  compute_log_numberic_features(df, log_numberic_features)
  #generate_sum_features(df)
  #generate_has_features(df)
  drop_converted_features(df, converted_features)
  _preview_examples(df)

  export_train_test_dataframes(train_df, test_df)


def main_titianic():

  fill_nan_value(df)
  bucketize_continuous_features(df)
  onehot_encode_integer_features(df)
  onehot_encode_string_features(df)
  generate_new_features(df)
  drop_features(df)

  drop_id_feature_for_training(df)
  print_final_features(df)
  export_file_path = "./train_fe1.csv"
  export_to_file(df, export_file_path)

  test_csv_file_path = "../data/test.csv"
  df = pd.read_csv(test_csv_file_path)
  fill_nan_value(df)
  bucketize_continuous_features(df)
  onehot_encode_integer_features(df)
  onehot_encode_string_features(df)
  generate_new_features(df)
  drop_features(df)
  #drop_id_feature_for_training(df)
  print_final_features(df)
  export_file_path = "./test_fe1.csv"
  export_to_file(df, export_file_path)


def _preview_examples(df, example_number=3):
  examples = df.head(example_number)
  print("Preview {} samples of the dataset".format(example_number))
  print(examples)


def _preview_examples_of_features(df, feature_name_array, example_number=3):
  examples = df[feature_name_array].head(example_number)
  print("Preview {} samples of the features: {}".format(
      example_number, feature_name_array))
  print(examples)


def load_train_test_dataframes():
  print("\n[Debug] Load the train and test files: ")

  train_csv_file_path = "./data/train.csv"
  test_csv_file_path = "./data/test.csv"
  train_df = pd.read_csv(train_csv_file_path)
  test_df = pd.read_csv(test_csv_file_path)
  print("Read train file: {} and test file: {}".format(train_csv_file_path,
                                                       test_csv_file_path))

  return train_df, test_df


def select_features(df, freature_name_array):
  print("\n[Debug] Select the features: ")
  print("Select the feature: {}".format(freature_name_array))
  return df[freature_name_array]


def drop_useless_features(df, feature_name_array):
  print("\n[Debug] Drop the useless features: ")
  print("Drop the features: {}".format(feature_name_array))
  df.drop(feature_name_array, axis=1, inplace=True)


def drop_missing_values_features(df, feature_name_array):
  print("\n[Debug] Drop the features of missing values: ")
  print("Drop the features: {}".format(feature_name_array))
  df.drop(feature_name_array, axis=1, inplace=True)
  _preview_examples(df)


def drop_converted_features(df, feature_name_array):
  print("\n[Debug] Drop the converted features: ")
  print("Drop the features: {}".format(feature_name_array))
  df.drop(feature_name_array, axis=1, inplace=True)


def _fill_nan_value_with_most(df, feature_name):
  value_count_series = df[feature_name].value_counts()
  most_value = value_count_series.index[0]
  print("For feature {}, fill with the most value: {}".format(
      feature_name, most_value))
  df[feature_name].fillna(most_value, inplace=True)


def _fill_nan_value_with_mean(df, feature_name):
  mean = df[feature_name].mean()
  print("For feature {}, fill with the mean value: {}".format(
      feature_name, mean))
  df.loc[df[feature_name].isnull(), feature_name] = mean


def _fill_nan_value_with_zero(df, feature_name):
  #df[feature_name] = df[feature_name].fillna("None")
  df[feature_name] = df[feature_name].fillna(0)


def fill_nan_value(df, fill_most_features, fill_mean_features,
                   fill_zero_features):
  print("\n[Debug] Fill the NaN values of the dataset: ")

  print("Print the NaN data of the dataset: ")
  null_feature_info = df.isnull().sum()
  print(null_feature_info)

  for feature_name in fill_most_features:
    _fill_nan_value_with_most(df, feature_name)

  for feature_name in fill_mean_features:
    _fill_nan_value_with_mean(df, feature_name)

  for feature_name in fill_zero_features:
    _fill_nan_value_with_zero(df, feature_name)

  _preview_examples_of_features(df, fill_most_features + fill_mean_features)

  print("\n[Debug] Print the NaN data of the dataset: ")
  null_feature_info = df.isnull().sum()
  print(null_feature_info)

  if null_feature_info.sum() == 0:
    print("Success: there is no feature with missing values")
  else:
    print("Error: there are {} features with missing values".format(
        null_feature_info.sum()))


def compute_log_numberic_features(df, feature_name_array):
  print("\n[Debug] Compute log of the numberic features: ")

  print("Compute the log of the features: {}".format(feature_name_array))
  for feature_name in feature_name_array:
    df[feature_name] = np.log1p(df[feature_name].values)


def bucketize_continuous_features(df):
  print("\n[Debug] Bucketize the continuous features: ")

  feature_name = "Age"
  bucket_number = 5
  min = df[feature_name].min()
  max = df[feature_name].max()
  interval = (max - min) / bucket_number
  # Example: [16.0, 32.0, 48.0, 64.0]
  buckets = [min + i * interval for i in range(1, bucket_number)]
  print("For feature {}, bucketize with number {} and the buckets {}".format(
      feature_name, bucket_number, buckets))

  new_feature_name_array = []
  for i in range(bucket_number):
    new_feature_name = "{}_{}".format(feature_name, i)
    new_feature_name_array.append(new_feature_name)
    df[new_feature_name] = 0

  df.loc[(df[feature_name] <= buckets[0]), new_feature_name_array[0]] = 1
  df.loc[(df[feature_name] > buckets[0]) & (df[feature_name] <= buckets[1]),
         new_feature_name_array[1]] = 1
  df.loc[(df[feature_name] > buckets[1]) & (df[feature_name] <= buckets[2]),
         new_feature_name_array[2]] = 1
  df.loc[(df[feature_name] > buckets[2]) & (df[feature_name] <= buckets[3]),
         new_feature_name_array[3]] = 1
  df.loc[(df[feature_name] > buckets[3]), new_feature_name_array[4]] = 1
  preview_examples_of_features(df, new_feature_name_array)

  feature_name = "Fare"
  bucket_number = 5
  min = df[feature_name].min()
  max = df[feature_name].max()
  interval = (max - min) / bucket_number
  # buckets = [min + i * interval for i in range(1, bucket_number)]
  # Use quantile-based with `pd.qcut(df['Fare'], 4)`
  buckets = [7.854, 10.5, 21.679, 39.688]
  print("For feature {}, bucketize with number {} and the buckets {}".format(
      feature_name, bucket_number, buckets))

  new_feature_name_array = []
  for i in range(bucket_number):
    new_feature_name = "{}_{}".format(feature_name, i)
    new_feature_name_array.append(new_feature_name)
    df[new_feature_name] = 0

  df.loc[(df[feature_name] <= buckets[0]), new_feature_name_array[0]] = 1
  df.loc[(df[feature_name] > buckets[0]) & (df[feature_name] <= buckets[1]),
         new_feature_name_array[1]] = 1
  df.loc[(df[feature_name] > buckets[1]) & (df[feature_name] <= buckets[2]),
         new_feature_name_array[2]] = 1
  df.loc[(df[feature_name] > buckets[2]) & (df[feature_name] <= buckets[3]),
         new_feature_name_array[3]] = 1
  df.loc[(df[feature_name] > buckets[3]), new_feature_name_array[4]] = 1
  preview_examples_of_features(df, new_feature_name_array)


def onehot_encode_integer_features(df):
  print(
      "\n[Debug] One-hot encode the integer features with ont-hot encoding: ")
  feature_name = "Pclass"
  values_range = [1, 2, 3]

  new_feature_name_array = []
  for i in range(len(values_range)):
    new_feature_name = "{}_{}".format(feature_name, i)
    new_feature_name_array.append(new_feature_name)
    df[new_feature_name] = 0
  print("For feature {}, replace {} with new features: {}".format(
      feature_name, values_range, new_feature_name_array))

  df.loc[(df[feature_name] == values_range[0]), new_feature_name_array[0]] = 1
  df.loc[(df[feature_name] == values_range[1]), new_feature_name_array[1]] = 1
  df.loc[(df[feature_name] == values_range[2]), new_feature_name_array[2]] = 1
  preview_examples_of_features(df, new_feature_name_array)


def _onehot_encode_string_feature(df, feature_name, unique_value_array):
  new_feature_name_array = []

  for i in range(len(unique_value_array)):
    new_feature_name = "{}_{}".format(feature_name, i)
    new_feature_name_array.append(new_feature_name)
    df[new_feature_name] = 0
    df.loc[(df[feature_name] == unique_value_array[i]), new_feature_name] = 1

  print("For feature {}, encode values: {} with new features: {}".format(
      feature_name, unique_value_array, new_feature_name_array))
  _preview_examples_of_features(df, new_feature_name_array)


def onehot_encode_string_features_for_train(df, string_features):
  print("\n[Debug] One-hot encode the string features for train dataset: ")

  # Example: {"Name": ["tobe", "wawa"]}
  feature_name_unique_value_map = {}
  for feature_name in string_features:
    unique_value_array = df[feature_name].unique()
    _onehot_encode_string_feature(df, feature_name, unique_value_array)
    feature_name_unique_value_map[feature_name] = unique_value_array

  return feature_name_unique_value_map


def onehot_encode_string_features_for_test(df, string_features,
                                           feature_name_unique_value_map):
  print("\n[Debug] One-hot encode the string features for test dataset: ")

  for feature_name in string_features:
    # Get the ["tobe", "wawa"] from {"Name": ["tobe", "wawa"]}
    unique_value_array = feature_name_unique_value_map[feature_name]
    _onehot_encode_string_feature(df, feature_name, unique_value_array)

  return feature_name_unique_value_map
  """
  feature_name = "Sex"
  values_range = ["male", "female"]
  new_feature_name_array = []
  for i in range(len(values_range)):
    new_feature_name = "{}_{}".format(feature_name, i)
    new_feature_name_array.append(new_feature_name)
    df[new_feature_name] = 0
  print("For feature {}, replace {} with new features: {}".format(feature_name, values_range, new_feature_name_array))

  df.loc[(df[feature_name] == values_range[0]), new_feature_name_array[0]] = 1
  df.loc[(df[feature_name] == values_range[1]), new_feature_name_array[1]] = 1
  preview_examples_of_features(df, new_feature_name_array)

  feature_name = "Embarked"
  values_range = ["S", "C", "Q"]
  new_feature_name_array = []
  for i in range(len(values_range)):
    new_feature_name = "{}_{}".format(feature_name, i)
    new_feature_name_array.append(new_feature_name)
    df[new_feature_name] = 0
  print("For feature {}, replace {} with new features: {}".format(feature_name, values_range, new_feature_name_array))

  df.loc[(df[feature_name] == values_range[0]), new_feature_name_array[0]] = 1
  df.loc[(df[feature_name] == values_range[1]), new_feature_name_array[1]] = 1
  df.loc[(df[feature_name] == values_range[2]), new_feature_name_array[2]] = 1
  preview_examples_of_features(df, new_feature_name_array)
  """


def generate_sum_features(df):
  print("\n[Debug] Generate the new features for sum relation: ")

  new_feature_name_array = ["TotalSF"]
  for feature_name in new_feature_name_array:
    df[feature_name] = 0

  df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
  """
  new_feature_name_array = ["Family_Size", "Alone"]
  print("Add the new features {}".format(new_feature_name_array))
  for feature_name in new_feature_name_array:
    df[feature_name] = 0

  df["Family_Size"] = df["Parch"] + df["SibSp"]

  df.loc[(df["Family_Size"] == 0), "Alone"] = 1
  """

  _preview_examples_of_features(df, new_feature_name_array)


def generate_has_features(df):
  print("\n[Debug] Generate the new features for has relation: ")

  new_feature_name_array = [
      "HasBasement", "HasGarage", "Has2ndFloor", "HasMasVnr", "HasWoodDeck",
      "HasPorch", "HasPool", "IsNew"
  ]
  for feature_name in new_feature_name_array:
    df[feature_name] = 0

  df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

  df.loc[(df["TotalBsmtSF"] > 0), "HasBasement"] = 1
  df.loc[(df["GarageArea"] > 0), "HasGarage"] = 1
  df.loc[(df["2ndFlrSF"] > 0), "Has2ndFloor"] = 1
  df.loc[(df["MasVnrArea"] > 0), "HasMasVnr"] = 1
  df.loc[(df["WoodDeckSF"] > 0), "HasWoodDeck"] = 1
  df.loc[(df["OpenPorchSF"] > 0), "HasPorch"] = 1
  df.loc[(df["PoolArea"] > 0), "HasPool"] = 1
  df.loc[(df["YearBuilt"] > 0), "IsNew"] = 1

  _preview_examples_of_features(df, new_feature_name_array)


def drop_string_features(df, feature_name_array):
  print("\n[Debug] Drop the string features: ")

  print("Drop string features: {}".format(feature_name_array))
  df.drop(feature_name_array, axis=1, inplace=True)


def drop_id_feature(df, id_feature_name):
  print("\n[Debug] Drop the id feature for training: ")
  print("Drop id feature: {}".format(id_feature_name))
  df.drop(id_feature_name, axis=1, inplace=True)
  """
  id_feature = ["PassengerId"]
  print("Drop the id features: {}".format(id_feature))
  df.drop(id_feature, axis=1, inplace=True)
  """

  print("\n[Debug] Check the final features: ")
  print("\nPlease check if it has not-numberic features")
  info = df.info()

  null_feature_info = df.isnull().sum()
  print("\nPlese check if it has null values")
  print(null_feature_info)

  print("\nPlease the final example")
  samples = df.head(1)
  print(samples)


def export_train_test_dataframes(train_df, test_df):
  export_file_path = "./generated/train_feature.csv"
  _export_dataframe_to_file(train_df, export_file_path)

  export_test_file_path = "./generated/test_feature.csv"
  _export_dataframe_to_file(test_df, export_test_file_path)


def _export_dataframe_to_file(df, export_file_path):
  print("\n[Debug] Export the dataset to the file: ")
  print("Export the csv file in {}".format(export_file_path))
  df.to_csv(export_file_path, index=False)


if __name__ == "__main__":
  #smoke_main()
  main()

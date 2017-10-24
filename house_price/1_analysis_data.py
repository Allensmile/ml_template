#!/usr/bin/env python

import numpy as np
import scipy as sp
import pandas as pd
import pprint


def main():
  train_df, test_df = load_dataframes()
  view_sample_dataset(train_df)
  print_dataset_info(train_df)
  print_features_number(train_df)
  print_feature_statistics(train_df)
  print_features_presence(train_df)
  print_features_presence(test_df)
  print_feature_topk_values(train_df)


def load_dataframes():
  print("\n[Debug] Load the dataset file: ")

  train_file_path = "./data/train.csv"
  test_file_path = "./data/test.csv"
  train_df = pd.read_csv(train_file_path)
  test_df = pd.read_csv(test_file_path)
  print("Read the train file: {} and test file: {}".format(
      train_file_path, test_file_path))
  return train_df, test_df


def view_sample_dataset(df, example_number=3):
  print("\n[Debug] View the sample of the dataset: ")

  print("Print the first {} examples".format(example_number))
  df_samples = df.head(example_number)
  print(df_samples)


def print_dataset_info(dataset):
  print("\n[Debug] Print the basic information of the dataset: ")

  example_number = len(dataset)
  print(example_number)
  example_number_threshold = 10000
  if example_number >= example_number_threshold:
    print("Has more than {} examples, consider using complex models".format(
        example_number_threshold))
  else:
    print("Has less than {} examples, consider using simple models".format(
        example_number_threshold))

  print("\n[Debug] Print the info of the dataset: ")
  dataset_info = dataset.info()
  print(dataset_info)


def print_features_number(dataset):
  print("\n[Debug] Print the number of the features: ")

  features_and_types = dataset.dtypes
  all_feature_name_array = []
  numberic_feature_name_array = []
  not_numberic_feature_name_array = []

  for i in range(len(features_and_types)):
    feature_type = features_and_types[i]
    feature_name = features_and_types.index[i]
    all_feature_name_array.append(feature_name)
    if feature_type == np.int16 or feature_type == np.int32 or feature_type == np.int64 or feature_type == np.float16 or feature_type == np.float32 or feature_type == np.float64 or feature_type == np.float128 or feature_type == np.double:
      numberic_feature_name_array.append(feature_name)
    else:
      not_numberic_feature_name_array.append(feature_name)

  print("Total feature number: {}".format(len(all_feature_name_array)))
  print(all_feature_name_array)
  print("Numberic feature number: {}".format(len(numberic_feature_name_array)))
  print(numberic_feature_name_array)
  print("Not numberic feature number: {}".format(
      len(not_numberic_feature_name_array)))
  print(not_numberic_feature_name_array)

  print("Print the feature list of the dataset: ")
  print(features_and_types)


def print_feature_statistics(df):
  print("\n[Debug] For numberic features, print the feature statistics: ")

  feature_statistics = df.describe()
  print(feature_statistics)


def print_features_presence(dataset):
  print("\n[Debug] Print the presence of the dataset: ")

  example_number = len(dataset)
  features_array = list(dataset.columns.values)
  # Example: [("Id", 1.0), ("Age", 0.78)]
  feature_name_presence_tuple_array = []
  for feature_name in features_array:
    #feature_presence_number = len(dataset[feature_name][dataset[feature_name].notnull()])
    feature_presence_number = dataset[feature_name].notnull().sum()
    feature_presence_percentage = 100.0 * feature_presence_number / example_number
    feature_name_presence_tuple_array.append((feature_name,
                                              feature_presence_percentage))
    # Example: "Age: 80.1346801347% (714 / 891)"
    print("{}: {}% ({} / {})".format(feature_name, feature_presence_percentage,
                                     feature_presence_number, example_number))

  throw_away_presence_threshold = 90
  # Example: ["Age_87%", "Sex_32%"]
  should_throw_away_feature_name_array = [
      tuple[0] for tuple in feature_name_presence_tuple_array
      if tuple[1] < throw_away_presence_threshold
  ]
  should_throw_away_feature_string_array = [
      "{} {}%".format(tuple[0], tuple[1])
      for tuple in feature_name_presence_tuple_array
      if tuple[1] < throw_away_presence_threshold
  ]
  print("Should throw away the number of features: {}".format(
      len(should_throw_away_feature_string_array)))
  print(should_throw_away_feature_name_array)
  pprint.pprint(should_throw_away_feature_string_array)
  should_fill_missing_feature_name_array = [
      tuple[0] for tuple in feature_name_presence_tuple_array
      if tuple[1] > throw_away_presence_threshold and tuple[1] < 100
  ]
  should_fill_missing_feature_string_array = [
      "{} {}%".format(tuple[0], tuple[1])
      for tuple in feature_name_presence_tuple_array
      if tuple[1] > throw_away_presence_threshold and tuple[1] < 100
  ]
  print("Should fill missing number of features: {}".format(
      len(should_fill_missing_feature_string_array)))
  print(should_fill_missing_feature_name_array)
  pprint.pprint(should_fill_missing_feature_string_array)


def print_feature_topk_values(dataset, top_k_number=5):
  print("\n[Debug] For all features, print the top-k values: ")

  features_array = list(dataset.columns.values)

  for i in range(len(features_array)):
    feature_name = features_array[i]
    # TODO: For some numberic function, show more thant k values
    #top_k_feature_info = dataset[feature_name].value_counts()[:top_k_number]
    top_k_feature_info = dataset[feature_name].value_counts()
    print("\nFeature {} and the top {} values:".format(feature_name,
                                                       top_k_number))
    print(top_k_feature_info)

    # TODO: Print the value range and the value distribution for each feature


if __name__ == "__main__":
  main()

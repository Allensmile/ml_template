#!/usr/bin/env python

import pandas as pd


def main():
  train_df, test_df = load_train_test_files()
  view_sample_dataset(train_df)
  view_sample_dataset(test_df)
  check_not_numberic_features(train_df)
  check_nan_values(train_df)
  check_not_numberic_features(test_df)
  check_nan_values(test_df)
  check_feature_number(train_df, test_df)


def load_train_test_files():
  print("\n[Debug] Load the train and test files: ")

  train_csv_file_path = "./generated/train_feature.csv"
  test_csv_file_path = "./generated/test_feature.csv"
  train_df = pd.read_csv(train_csv_file_path)
  test_df = pd.read_csv(test_csv_file_path)
  print("Read train file: {} and test file: {}".format(train_csv_file_path,
                                                       test_csv_file_path))
  return train_df, test_df


def view_sample_dataset(df, example_number=3):
  print("\n[Debug] View the sample of the dataset: ")

  print("Print the first {} examples".format(example_number))
  df_samples = df.head(example_number)
  print(df_samples)


def check_not_numberic_features(df):
  print("\n[Debug] Please check if it has not-numberic features")
  info = df.info()


def check_nan_values(df):
  print("\n[Debug] Please check if it has NaN values")

  nan_value_number = df.isnull().values.sum()
  if nan_value_number == 0:
    print("Success: there is no nan value")
  else:
    print("Error: there are {} nan values".format(nan_value_number))


def check_feature_number(train_df, test_df):
  print("\n[Debug] Check feature number of train and test dataset: ")

  # Remove the id column and label column in train dataset
  train_feature_number = len(train_df.count()) - 2
  # Remove the id column in test dataset
  test_feature_number = len(test_df.count()) - 1
  print("The feature number of train file: {} and test file: {}".format(
      train_feature_number, test_feature_number))
  if train_feature_number == test_feature_number:
    print("Success: the feature numbers match")
  else:
    print("Error: the feature numbers mismatch")


if __name__ == "__main__":
  main()

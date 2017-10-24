#!/usr/bin/env python

import numpy as np
import scipy as sp
import pandas as pd
import pprint


def main():
  sample_submission_df, my_submission_df = load_submission_dataframes()
  preview_submission_files(sample_submission_df, my_submission_df)
  check_line_number(sample_submission_df, my_submission_df)
  check_id_colume(sample_submission_df, my_submission_df)


def load_submission_dataframes():
  print("\n[Debug] Load the files: ")

  sample_submission_file_path = "./data/sample_submission.csv"
  my_submission_file_path = "./generated/my_submission.csv"
  sample_submission_df = pd.read_csv(sample_submission_file_path)
  my_submission_df = pd.read_csv(my_submission_file_path)
  print("Read the sample submission file: {} and my submission file: {}".
        format(sample_submission_file_path, my_submission_file_path))

  return sample_submission_df, my_submission_df


def preview_submission_files(sample_submission_df, my_submission_df):
  print("\n[Debug] Preview the submission files: ")

  example_number = 5
  sample_submission_examples = sample_submission_df.head(example_number)
  print(
      "Preview {} examples from the sample submission".format(example_number))
  print(sample_submission_examples)

  my_submission_examples = my_submission_df.head(example_number)
  print("Preview {} examples from my submission".format(example_number))
  print(my_submission_examples)


def check_line_number(expected_df, actual_df):
  print("\n[Debug] Check the line number of the files: ")

  expected_line_number = len(expected_df)
  actual_line_number = len(actual_df)

  print("The line numbers of expected: {}, actual: {}".format(
      expected_line_number, actual_line_number))
  if expected_line_number == actual_line_number:
    print("Success: the line numbers match")
  else:
    print("Error: the line numbers mismatch")


def check_id_colume(expected_df, actual_df):
  print("\n[Debug] Check the id colume of the files: ")
  id_column_name = "Id"
  mismatch_number = len(expected_df.loc[expected_df[id_column_name] !=
                                        actual_df[id_column_name]])

  if mismatch_number > 0:
    print("Error: the id columes mismatch for number: {}".format(
        mismatch_number))
  else:
    print("Success: the id columes match")


def check_column_number(expected_df, actual_df):
  print("\n[Debug] Check column number of the files: ")

  expected_column_number = len(expected_df.count())
  actual_column_number = len(actual_df.count())
  print("The column number of expected: {}, actual: {}".format(
      expected_column_number, actual_column_number))
  if expected_column_number == actual_column_number:
    print("Success: the column number match")
  else:
    print("Error: the column number mismatch")


if __name__ == "__main__":
  main()

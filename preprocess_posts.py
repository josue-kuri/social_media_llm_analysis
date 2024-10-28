
from constants import *

import argparse
import glob
import logging
import pandas as pd


pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 80)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 180)

def load_posts(input_data_path):
    csv_profiles = glob.glob(input_data_path, recursive=True)
    acc = []
    for f in csv_profiles:
        source = f.split('/')[-2]
        df_tmp = pd.read_csv(
            f,
            usecols=twitter_columns,
            parse_dates=['created_at']
        )
        df_tmp.rename(columns={'full_text': 'text'}, inplace=True)
        df_tmp['author'] = source
        acc.append(df_tmp)
    df = pd.concat(acc).drop_duplicates().reset_index(drop=True)

    return df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_data_path",
        default=None,
        type=str,
        required=True,
        help="Path to scrapped Twitter posts."
    )
    parser.add_argument(
        "--output_file_name",
        default=None,
        type=str,
        required=True,
        help="Name of parquet file with processed posts."
    )
    args = parser.parse_args()

    df_posts = load_posts(args.input_data_path)

    print(df_posts.head())
    print(df_posts.tail())

    df_posts.to_parquet(args.output_file_name, index=False)


if __name__ == "__main__":
    main()

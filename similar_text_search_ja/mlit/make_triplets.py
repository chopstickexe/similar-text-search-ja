from pathlib import Path
from random import randrange
from typing import Set

import numpy as np
import pandas as pd
from similar_text_search_ja.mlit import config
from sklearn.model_selection import train_test_split


def __get_documents(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, header=0)


def __sattolo_cycle(items):
    items = list(items)
    i = len(items)
    while i > 1:
        i -= 1
        j = randrange(i)  # 0 <= j <= i-1
        items[j], items[i] = items[i], items[j]  # Switch
    return items


def __outside_group_rand(items, ids: Set[int]):
    return np.random.choice(list(ids.difference(items)), size=len(items), replace=False)


def __generate_triplets(
    df: pd.DataFrame,
    group_col: str,
    txt_col: str,
    t_txt_col: str,
    p_txt_col: str,
    n_txt_col: str,
):
    # https://stackoverflow.com/questions/57931505/pandas-sample-based-on-category-for-each-row
    id_col = "id"
    p_id_col = "pid"
    n_id_col = "nid"

    df[id_col] = df.index.to_series()
    df[p_id_col] = df.groupby(group_col)[id_col].transform(__sattolo_cycle)
    df[n_id_col] = df.groupby(group_col)[id_col].transform(
        __outside_group_rand, set(df[id_col])
    )

    ret_df = df[[txt_col, p_id_col, n_id_col]].rename(columns={txt_col: t_txt_col})
    ret_df = ret_df.join(df[txt_col], on=p_id_col).rename(columns={txt_col: p_txt_col})
    ret_df = ret_df.join(df[txt_col], on=n_id_col).rename(columns={txt_col: n_txt_col})
    return ret_df[[t_txt_col, p_txt_col, n_txt_col]]


def __split_train_dev_test(
    df: pd.DataFrame, train_size: float, dev_size: float, test_size: float
):
    train, dev_test = train_test_split(df, train_size=train_size, random_state=4)
    dev, test = train_test_split(
        dev_test, train_size=dev_size / (dev_size + test_size), random_state=7
    )
    return train, dev, test


def __create_dataset_dir(dataset_dir: Path):
    dataset_dir.mkdir(parents=True, exist_ok=True)


def __to_tsv(df: pd.DataFrame, csv_path: str):
    df.to_csv(csv_path, header=False, index=False, sep="\t")


def main():
    conf = config.get_config()
    mlit_conf = conf["mlit"]
    txt_col = mlit_conf["target_fields"][0]  # TODO Should concat all the target fields
    p_txt_col = "p"
    n_txt_col = "n"
    t_txt_col = "t"  # target text

    dir_path = Path(mlit_conf["triplet_dataset_dir"])
    __create_dataset_dir(dir_path)

    df = __get_documents(mlit_conf["raw_csv_path"])
    df = __generate_triplets(
        df, mlit_conf["ans_field"], txt_col, t_txt_col, p_txt_col, n_txt_col
    )
    train, dev, test = __split_train_dev_test(df, 0.4, 0.3, 0.3)

    __to_tsv(train, str(dir_path / "train.tsv"))
    __to_tsv(dev, str(dir_path / "dev.tsv"))
    __to_tsv(test, str(dir_path / "test.tsv"))


if __name__ == "__main__":
    main()

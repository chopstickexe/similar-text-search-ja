from random import randrange
from typing import List, Set

import numpy as np
import pandas as pd
from similar_text_search_ja.mlit import config


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
    df: pd.DataFrame, id_col: str, p_id_col: str, n_id_col: str, group_col: str
):
    # https://stackoverflow.com/questions/57931505/pandas-sample-based-on-category-for-each-row
    df[p_id_col] = df.groupby(group_col)[id_col].transform(__sattolo_cycle)
    df[n_id_col] = df.groupby(group_col)[id_col].transform(
        __outside_group_rand, set(df[id_col])
    )


def __to_tsv(df: pd.DataFrame, csv_path: str, columns: List[str]):
    df.to_csv(csv_path, header=False, index=False, columns=columns, sep="\t")


def main():
    conf = config.get_config()
    mlit_conf = conf["mlit"]
    id_col = "id"
    txt_col = mlit_conf["target_fields"][0]  # TODO Should concat all the target fields
    p_id_col = "pid"
    n_id_col = "nid"
    p_txt_col = "p"
    n_txt_col = "n"
    t_txt_col = "t"  # target text

    df = __get_documents(mlit_conf["csv_path"])
    df[id_col] = df.index.to_series()

    __generate_triplets(df, id_col, p_id_col, n_id_col, mlit_conf["ans_field"])
    ret_df = df[[txt_col, p_id_col, n_id_col]].rename(columns={txt_col: t_txt_col})
    ret_df = ret_df.join(df[txt_col], on=p_id_col).rename(columns={txt_col: p_txt_col})
    ret_df = ret_df.join(df[txt_col], on=n_id_col).rename(columns={txt_col: n_txt_col})
    __to_tsv(ret_df, mlit_conf["triplet_tsv_path"], [t_txt_col, p_txt_col, n_txt_col])


if __name__ == "__main__":
    main()

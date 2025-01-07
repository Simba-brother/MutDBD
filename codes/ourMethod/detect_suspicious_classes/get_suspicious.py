from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import pandas as pd
import scikit_posthocs as sp
from collections import defaultdict

def fine_tune_columns(df:pd.DataFrame):
    new_df = pd.DataFrame()
    columns = df.columns
    for col in columns:
        if df[col].nunique() == 1:
            value_list = list(df[col])
            value_list[-1] += 1e-10
            new_df[col] = value_list
        else:
            new_df[col] = list(df[col])
    return new_df



def get_suspicious_classes_by_ScottKnottESD(data_dict):
    # https://github.com/klainfo/ScottKnottESD
    '''
    data_dict:{Int(class_idx):list(precision|recall|F1)}
    '''
    pandas2ri.activate()
    sk = importr("ScottKnottESD")
    df = pd.DataFrame(data_dict)
    new_df = fine_tune_columns(df)

    r_sk = sk.sk_esd(new_df)
    column_order = [x-1 for x in list(r_sk[3])]

    ranking = pd.DataFrame(
        {
            "Class": [new_df.columns[i] for i in column_order],
            "rank": r_sk[1].astype("int"),
        })
    Class_list = list(ranking["Class"])
    rank_list = list(ranking["rank"])
    group_map = defaultdict(list)
    for class_idx, rank in zip(Class_list,rank_list):
        group_map[rank].append(class_idx)
    group_key_list = list(group_map.keys())
    group_key_list.sort() # replace
    top_key = group_key_list[0]
    low_key = group_key_list[-1]
    suspicious_classes_top = group_map[top_key]
    suspicious_classes_low = group_map[low_key]
    return suspicious_classes_top,suspicious_classes_low

if __name__ == "__main__":
    data_dict = {
        0:[4,4,4,4],
        1:[4,4,4,4],
    }
    get_suspicious_classes_by_ScottKnottESD(data_dict)
    # result = sp.posthoc_ske(data_dict)

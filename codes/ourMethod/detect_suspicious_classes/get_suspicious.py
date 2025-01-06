from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import pandas as pd
from collections import defaultdict

def get_suspicious_classes_by_ScottKnottESD(data_dict):
    # https://github.com/klainfo/ScottKnottESD
    '''
    data_dict:{Int(class_idx):list(precision|recall|F1)}
    '''
    pandas2ri.activate()
    sk = importr("ScottKnottESD")
    df = pd.DataFrame(data_dict)

    r_sk = sk.sk_esd(df)
    column_order = [x-1 for x in list(r_sk[3])]

    ranking = pd.DataFrame(
        {
            "Class": [df.columns[i] for i in column_order],
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
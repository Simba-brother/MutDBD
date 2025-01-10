from sklearn.metrics import log_loss
import pandas as pd
import joblib
import os
from collections import defaultdict
from codes import config

def get_cross_entropy_loss(y_true,y_pred,labels):
    return log_loss(y_true,y_pred,labels=labels)


def get_class_dict(df):
    data_dict = {}
    labels = list(range(config.class_num))
    m_i_list = list(range(500))
    for class_i in range(config.class_num):
        df_class_i = df.loc[df["GT_label"]==class_i]
        y_true = list(df_class_i["GT_label"])
        avg_CE_loss = 0
        for m_i in m_i_list:
            y_pred = []
            prob_output_list = (df_class_i[f"model_{m_i}"])
            for prob_output in prob_output_list:
                prob_output = list(prob_output)
                prob_output = [round(prob,4) for prob in prob_output]
                prob_output[-1] = 1.0-sum(prob_output[:-1])
                y_pred.append(prob_output)
            # model_i在class_i数据集的上的交叉熵loss
            m_i_CEloss = get_cross_entropy_loss(y_true,y_pred,labels)
            avg_CE_loss += m_i_CEloss
        avg_CE_loss /= len(m_i_list)
        avg_CE_loss = round(avg_CE_loss,4)
        data_dict[class_i] = avg_CE_loss
    return data_dict



def main():
    data_dict = {}
    for rate in config.fine_mutation_rate_list:
        prob_outputs_data_path = os.path.join(
            config.exp_root_dir,
            "EvalMutationToCSV",
            config.dataset_name,
            config.model_name,
            config.attack_name,
            str(rate),
            "prob_outputs.parquet"
        )
        '''
        prob_outputs = {
            model_id:[prob_outputs_0,prob_outputs_1,...],
            "sampled_id":[0,1,2,...]
            "GT_label":[0,0,..,1,1,..,9,]
            "isPoisoned":[True,False,...]
        }
        '''
        df = pd.read_parquet(prob_outputs_data_path)
        class_dict = get_class_dict(df)
        data_dict[rate]=class_dict
    new_class_dict = defaultdict(list)
    for class_i in range(config.class_num):
        for rate in config.fine_mutation_rate_list:
            new_class_dict[class_i].append(data_dict[rate][class_i])
    # 准备画图
    pass

    

if __name__ == "__main__":
    main()
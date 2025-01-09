from sklearn.metrics import log_loss
import pandas as pd
import joblib
import os
from codes import config
def get_cross_entropy_loss(y_true,y_pred,labels):
    log_loss(y_true,y_pred,labels)


def get_class_dict(prob_outputs):
    df = pd.DataFrame(prob_outputs)
    gt_label_list = prob_outputs["GT_label"]
    for m_i in range(500):
        cur_model_prob_output_list = prob_outputs[f"model_{m_i}"]
        for cur_model_prob_output,gt_label in zip(cur_model_prob_output_list,gt_label_list):
            pass

def main():
    # 加载confidence csv
    rate = 0.01
    prob_outputs_data_path = os.path.join(
        config.exp_root_dir,
        "EvalMutationToCSV",
        config.dataset_name,
        config.model_name,
        config.attack_name,
        str(rate),
        "prob_outputs.data"
    )
    '''
    prob_outputs = {
        model_id:[prob_outputs_0,prob_outputs_1,...],
        "sampled_id":[0,1,2,...]
        "GT_label":[0,0,..,1,1,..,9,]
        "isPoisoned":[True,False,...]
    }
    '''
    prob_outputs = joblib.load(prob_outputs_data_path)
    get_class_dict(prob_outputs)
    

if __name__ == "__main__":
    main()
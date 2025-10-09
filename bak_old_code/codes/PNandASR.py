'''
32个场景中PN和ASR假设检验关系

参考文献 Training Data Debugging for the Fairness of Machine Learning Software,(ICSE'22),Linghan Meng,Huiyan Li)
技术参考：
(1):https://zhuanlan.zhihu.com/p/22692029
(2):https://www.datascienceconcepts.com/tutorials/python-programming-language/omitted-variable-bias-wald-test-in-python/

'''
import os
import joblib
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr,pearsonr

def get_unit_data(dataset_name, model_name, attack_name):
    save_dir = os.path.join(exp_root_dir, "实验结果", dataset_name, model_name, attack_name)
    os.makedirs(save_dir,exist_ok=True)
    save_file_name = "res_1.pkl"  # res.pkl
    save_path = os.path.join(save_dir, save_file_name)
    res_dict = joblib.load(save_path)
    our_acc_list = res_dict["our_acc_list"]
    our_asr_list = res_dict["our_asr_list"]
    our_p_num_list = res_dict["our_p_num_list"]
    
    # asd_acc_list = res_dict["asd_acc_list"]
    # asd_asr_list = res_dict["asd_asr_list"]
    # asd_p_num_list = res_dict["asd_p_num_list"]

    avg_p_num = sum(our_p_num_list)/len(our_p_num_list)
    avg_asr = sum(our_asr_list)/len(our_asr_list)
    return avg_p_num, avg_asr

def get_data():

    p_num_list = []
    asr_list = []
    dataset_name_list = ["CIFAR10","GTSRB","ImageNet2012_subset"]
    model_name_list = ["ResNet18","VGG19","DenseNet"]
    attack_name_list = ["BadNets","IAD","Refool","WaNet"]
    for dataset_name in dataset_name_list:
        for model_name in model_name_list:
            if dataset_name == "ImageNet2012_subset" and model_name == "VGG19":
                continue
            for attack_name in attack_name_list:
                p_num,asr = get_unit_data(dataset_name,model_name,attack_name)
                p_num_list.append(p_num)
                asr_list.append(asr)
    return p_num_list,asr_list

def calcu(x_list,y_list):
    '''
    x_list:是自变量
    y_list:是因变量
    '''
    x_list =  sm.add_constant(x_list)
    model = sm.OLS(y_list, x_list) # 此处注意因变量在前。
    results = model.fit()
    wald_test = results.wald_test('x1 = 0')  # 检验X的系数是否为0
    return results, wald_test

def main():
    p_num_list, asr_list = get_data()
    results,wald_test = calcu(p_num_list,asr_list)
    print(results.summary())
    print("回归系数:")
    print(results.params)
    print("回归系数P值:")
    print(results.pvalues)
    print("wald_test:")
    print(wald_test)
    # 计算斯皮尔曼相关系数
    rho, p_value = spearmanr(p_num_list,asr_list)
    print(f"斯皮尔曼相关系数:{rho},P值:{p_value}")
    # 皮尔逊相关系数
    rho, p_value = pearsonr(p_num_list,asr_list)
    print(f"皮尔逊相关系数:{rho},P值:{p_value}")

if __name__ == "__main__":
    exp_root_dir = "/data/mml/backdoor_detect/experiments/"
    main()

    


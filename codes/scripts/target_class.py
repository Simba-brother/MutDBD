'''
第一步:确定target_class
'''

import sys
sys.path.append("./")
import os
import joblib
from collections import defaultdict
import numpy as np
from scipy import stats
from cliffs_delta import cliffs_delta
from codes.draw import draw_box
from codes.utils import create_dir
from codes import config


class TargetClassProcessor():
    def __init__(self,
                 dataset_name, # 数据集名称
                 model_name, # 模型名称
                 attack_name, # 攻击名称
                 mutation_name_list, # 变异算子名称list
                 mutation_rate_list, # 变异率list
                 exp_root_dir, # 实验数据根目录
                 class_num, # 数据集的分类数
                 mutated_model_num, # 每个变异率下变异模型的数量,eg:50
                 mutation_operator_num): # 变异算子的数量 = len(mutation_name_list),eg:5
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.attack_name = attack_name
        self.mutation_name_list = mutation_name_list
        self.mutation_rate_list = mutation_rate_list
        self.exp_root_dir = exp_root_dir
        self.class_num = class_num
        self.mutated_model_num = mutated_model_num 
        self.mutation_operator_num = mutation_operator_num

    def get_dict_of_each_mutation_rate_each_classes_precision_list_by_mutation_name(self,mutation_name):
        '''
        得到某个变异算子下 > 在不同变异率下 > 变异模型们在各个类别上的precision
        '''
        # 返回的数据
        ans_dic = {}
        # 数据结构初始化
        # {mutation_rate:{class_idx:[precision]}}
        for mutation_rate in self.mutation_rate_list:
            ans_dic[mutation_rate] = {}
            for class_idx in range(class_num):
                ans_dic[mutation_rate][class_idx] = []
        # 读取原始数据（在poisoned_trainset上的评估report）
        eval_poisoned_trainset_report_path = os.path.join(
            self.exp_root_dir,
            self.dataset_name,
            self,model_name,
            self.attack_name,
            mutation_name, 
            "eval_poisoned_trainset_report.data")
        eval_poisoned_trainset_report = joblib.load(eval_poisoned_trainset_report_path)
        # 遍历变异率
        for mutation_rate in self.mutation_rate_list:
            # 当前变异率下所有变异模型的评估报告（ mutation_operator_num * mutated_model_num)
            report_list = eval_poisoned_trainset_report[mutation_rate]
            # 遍历评估报告
            for report in report_list:
                # 当前评估报告，遍历class_idx
                for class_idx in range(self.class_num):
                    precision = report[str(class_idx)]["precision"]
                    ans_dic[mutation_rate][class_idx].append(precision)
        assert len(ans_dic[0.01][0]) == self.mutated_model_num, "数量不对" # 50
        return ans_dic

    def get_dict_of_each_mutation_rate_each_classes_precision_list_by_hybrid_mutator(self):
        '''
        所有变异算子混合 > 在不同变异率下 > 变异模型们在各个类别上的precision
        '''
        # 返回的数据
        data_dic = {}
        # 初始化数据结构
        # {mutation_rate:{class_idx:[precision]}}
        for mutation_rate in self.mutation_rate_list:
            data_dic[mutation_rate] = {}
            for class_idx in range(self.class_num):
                data_dic[mutation_rate][class_idx] = []
        # 遍历变异算子
        for mutation_name in self.mutation_name_list:
            eval_poisoned_trainset_report_path = os.path.join(
                self.exp_root_dir,
                self.dataset_name,
                self.model_name,
                self.attack_name, 
                mutation_name,
                "eval_poisoned_trainset_report.data")
            eval_poisoned_trainset_report = joblib.load(eval_poisoned_trainset_report_path)
            # 遍历变异率
            for mutation_rate in self.mutation_rate_list:
                # 当前变异率下所有变异模型的评估报告（ mutation_operator_num * mutated_model_num)
                report_list = eval_poisoned_trainset_report[mutation_rate]
                # 遍历评估报告
                for report in report_list:
                    # 当前评估报告，遍历class_idx
                    for class_idx in range(self.class_num):
                        precision = report[str(class_idx)]["precision"]
                        data_dic[mutation_rate][class_idx].append(precision)
        assert len(data_dic[0.01][0]) == self.mutated_model_num * self.mutation_operator_num, "数量不对" # 250
        return data_dic

    def get_target_class_with_dif_mutation_rate(self,data_dic):
        '''
        data_dic:
            数据结构：{mutation_rate:{class_idx:[precision]}}
            描述：即不同变异率下 > 各个类别上 > 预测精度list
        Return:
            ans_dic:
            数据结构：{mutation_rate:{"max_mean_value_class_idx":-1, "max_median_value_class_idx":-1, "target_class_idx":-1}}
            描述：即不同变异率下 > target class
        '''
        ans_dic = {}
        for mutation_rate in self.mutation_rate_list:
            ans_dic[mutation_rate] = {"max_mean_value_class_idx":-1, "max_median_value_class_idx":-1, "target_class_idx":-1}

        for mutation_rate in self.mutation_rate_list:
            max_mean_value = 0
            max_mean_value_class_idx = -1
            max_median_value = 0
            max_median_value_class_idx = -1
            for class_idx in range(self.class_num):
                mean_value = np.mean(data_dic[mutation_rate][class_idx])
                median_value = np.median(data_dic[mutation_rate][class_idx])
                if mean_value > max_mean_value:
                    max_mean_value = mean_value
                    max_mean_value_class_idx = class_idx
                if median_value > max_median_value:
                    max_median_value = median_value
                    max_median_value_class_idx = class_idx
            ans_dic[mutation_rate]["max_mean_value_class_idx"] = max_mean_value_class_idx
            ans_dic[mutation_rate]["max_median_value_class_idx"] = max_median_value_class_idx
            if max_mean_value_class_idx == max_median_value_class_idx:
                ans_dic[mutation_rate]["target_class_idx"] = max_mean_value_class_idx
        return ans_dic

    def get_pValue_Ciff_with_dif_mutation_rate(self, data_dict_1, data_dict_2):
        '''
        Args:
            data_dict_1:
                数据结构：{mutaion_rate:{class_idx:[precision]}}
            data_dict_2:
                数据结构：{mutaion_rate:{target_class_idx:_}}
        Return:
            res:
                数据结构：{mutation_rate:{
                            "target_class_i":target_class_idx,
                            "p_value_list":p_value_list,
                            "clif_delta_list":cliff_delta_list}
                        }
                描述：获得不同变异率下>target_class和其pvalue_list
        '''
        res = {}
        for mutation_rate in self.mutation_rate_list:
            target_class_idx = data_dict_2[mutation_rate]["target_class_idx"]
            if target_class_idx == -1:
                continue
            source_list = data_dict_1[mutation_rate][target_class_idx]
            other_list_list = []
            for class_idx in range(self.class_num):
                if class_idx == target_class_idx:
                    continue
                other_list_list.append(data_dict_1[mutation_rate][class_idx])
            p_value_list = []
            cliff_delta_list = []
            for other_list in other_list_list:
                if source_list == other_list:
                    p_value_list.append(float("inf"))
                    cliff_delta_list.append(0.0)
                    continue
                # 计算p值
                p_value = stats.wilcoxon(source_list, other_list).pvalue
                p_value_list.append(p_value)
                # 计算clif delta
                source_list_sorted = sorted(source_list)
                other_list_sorted = sorted(other_list)
                d,info = cliffs_delta(source_list_sorted, other_list_sorted)
                cliff_delta_list.append(abs(d))
            res[mutation_rate] = {
                "target_class_i":target_class_idx,
                "p_value_list":p_value_list,
                "clif_delta_list":cliff_delta_list
            }
        return res
    
    def get_AdaptiveRate_and_TargetClass(self,data_dict):
        '''
        data_dict:
            数据结构：{mutation_rate:{
                        "target_class_i":target_class_idx,
                        "p_value_list":p_value_list,
                        "clif_delta_list":cliff_delta_list}
                    }
        Return:
            adaptive_rate
            target_class_i
        '''
        adaptive_rate = -1
        target_class_i = -1
        candidate_mutation_ratio_list = sorted(list(data_dict.keys()))
        for mutation_ratio in candidate_mutation_ratio_list:
            p_value_list = data_dict[mutation_ratio]["p_value_list"]
            clif_delta_list = data_dict[mutation_ratio]["clif_delta_list"]
            all_P_flag = all(p_value < 0.05 for p_value in p_value_list)
            all_C_flag = all(d >= 0.147 for d in clif_delta_list)
            if all_P_flag is True and all_C_flag is True:
                adaptive_rate = mutation_ratio
                target_class_i = data_dict[mutation_ratio]["target_class_i"]
                break 
            

        
        return adaptive_rate, target_class_i

    def get_target_class_by_mutation_name(self,mutation_name):
        '''
        获得该变异算子下, 不同变异率下 > target class
        '''
        ans_dict_1 = self.get_dict_of_each_mutation_rate_each_classes_precision_list_by_mutation_name(mutation_name)
        ans_dict_2 = self.get_target_class_with_dif_mutation_rate(ans_dict_1)
        return ans_dict_2
    
    def get_target_class_by_hybrid_mutator(self):
        '''
        获得所有变异算子混合后，不同变异率下 > target_class
        '''
        ans_dict_1 = self.get_dict_of_each_mutation_rate_each_classes_precision_list_by_hybrid_mutator()
        ans_dict_2 = self.get_target_class_with_dif_mutation_rate(ans_dict_1)
        return (ans_dict_2)

    def get_adaptive_rate_of_Hybrid_mutator(self):
        '''
        得到混合变异算子，自适应变异率
        具体来说就是
            1:先将每个变异率下,所有的变异算子生成的变异模型进行混合,以此为每个变异率下所有的变异模型。
            2:根据变异模型们在各个类别上的精度表现,确定出变异率。
        Return:
            data_dict_4: {"adaptive_rate":adaptive_rate,  "target_class_i":target_class_i}
            data_dic_1:
                结构：{mutation_rate:{class_idx:[precision]}}
        '''
        data_dic_1 = self.get_dict_of_each_mutation_rate_each_classes_precision_list_by_hybrid_mutator()
        data_dic_2 = self.get_target_class_with_dif_mutation_rate(data_dic_1)
        ans_dict_3 = self.get_pValue_Ciff_with_dif_mutation_rate(data_dic_1, data_dic_2)
        adaptive_rate, target_class_i = self.get_AdaptiveRate_and_TargetClass(ans_dict_3)
        data_dict_4 = {"adaptive_rate":adaptive_rate,  "target_class_i":target_class_i}
        return data_dict_4, data_dic_1

    def get_adaptive_ratio_of_Combin_mutator(self):
        '''
        获得各个变异算子下>自适应变异率和target_class
        Return:
            ans:{mutation_name:
                    {"adaptive_rate":adaptive_rate,  "target_class_i":target_class_i}
                }
        '''
        ans = {}
        for mutation_name in mutation_name_list:
            dict_1 = self.get_dict_of_each_mutation_rate_each_classes_precision_list_by_mutation_name(mutation_name)
            dict_2 = self.get_target_class_with_dif_mutation_rate(dict_1)
            dict_3 = self.get_pValue_Ciff_with_dif_mutation_rate(dict_1, dict_2)
            adaptive_rate, target_class_i = self.get_AdaptiveRate_and_TargetClass(dict_3)
            if adaptive_rate == -1:
                classes_precision_dict = None
            else:
                classes_precision_dict = dict_1[adaptive_rate]
            ans[mutation_name] = {"adaptive_rate":adaptive_rate,  "target_class_i":target_class_i, "classes_precision_dict":classes_precision_dict}
        return ans

# def get_classes_precision_of_Hybrid_mutator_by_adpative_mutation_rate():
#     '''
#     获得混合变异算子在自适应变异率后其在各个class上precision list
#     Return:
#         ans:{class_idx:[precision]}
#     '''
#     data_dict_4, data_dic_1 = get_adaptive_rate_of_Hybrid_mutator()
#     adaptive_rate = data_dict_4["adaptive_rate"]
#     ans = {}
#     for class_idx in range(class_num):
#         ans[class_idx] = data_dic_1[adaptive_rate][class_idx]
#     # 绘图
#     save_dir = os.path.join(exp_root_dir, "images/box", dataset_name, model_name, attack_name, "Hybrid", "adaptive_rate")
#     create_dir(save_dir)
#     all_y = []
#     labels = []
#     for class_idx in range(class_num):
#         y_list = ans[class_idx]
#         all_y.append(y_list)
#         labels.append(f"Class_{class_idx}")
#     title = f"{dataset_name}_{model_name}_{attack_name}_Hybrid_adaptive_rate:{adaptive_rate}"
#     save_file_name = title+".png"
#     save_path = os.path.join(save_dir, save_file_name)
#     xlabel = "Category"
#     ylabel = "Precision"
#     draw_box(all_y, labels, title, xlabel, ylabel, save_path)
#     print(f"mutated_model_num:{mutated_model_num*mutated_operator_num}")
#     print("get_classes_precision_of_Hybrid_mutator_by_adpative_mutation_rate() success")
#     return ans



# def get_classes_precision_by_combin_mutator_with_adaptive_rate():
#     dic_1 = get_adaptive_ratio_of_Combin_mutator()
#     ans = defaultdict(list) # {class_idx:[precision]}
#     for class_idx in range(class_num):
#         for mutation_name in mutation_name_list:
#             classes_precision_dict = dic_1[mutation_name]["classes_precision_dict"]
#             if classes_precision_dict == None:
#                 continue
#             ans[class_idx].extend(classes_precision_dict[class_idx])
#     cur_mutated_model_num = len(ans[0])
#     if cur_mutated_model_num == 0:
#         print("没有一个变异算子自适应出合适的变异比例。。。。")
#         return None
#     return ans

# def get_target_class_by_combin_mutator_with_adaptive_rate():
#     '''
#     data_dic:{class_idx:[precision]}
#     Return:
#         ans_dic:{"max_mean_value_class_idx":-1, "max_median_value_class_idx":-1, "target_class_idx":-1}
#     ''' 
#     dic_1 = get_classes_precision_by_combin_mutator_with_adaptive_rate()
#     dic_ans = {"max_mean_value_class_idx":-1, "max_median_value_class_idx":-1, "target_class_idx":-1}
#     if dic_1 == None:
#         return dic_1, dic_ans
#     max_mean_value = 0
#     max_mean_value_class_idx = -1
#     max_median_value = 0
#     max_median_value_class_idx = -1
#     for class_idx in range(class_num):
#         mean_value = np.mean(dic_1[class_idx])
#         median_value = np.median(dic_1[class_idx])
#         if mean_value > max_mean_value:
#             max_mean_value = mean_value
#             max_mean_value_class_idx = class_idx
#         if median_value > max_median_value:
#             max_median_value = median_value
#             max_median_value_class_idx = class_idx
#     dic_ans["max_mean_value_class_idx"] = max_mean_value_class_idx
#     dic_ans["max_median_value_class_idx"] = max_median_value_class_idx
#     if max_mean_value_class_idx  == max_median_value_class_idx:
#         dic_ans["target_class_idx"] = max_median_value_class_idx
#     return dic_ans, dic_1

# def get_classes_precision_of_Combination_mutator_by_adpative_mutation_rate():
#     dic_1 = get_classes_precision_by_combin_mutator_with_adaptive_rate()
#     if dic_1 == None:
#         print("没有一个变异算子自适应出合适的变异比例。。。。")
#         return None
#     cur_mutated_model_num = len(dic_1[0])
#     # 绘图
#     save_dir = os.path.join(exp_root_dir, "images/box", dataset_name, model_name, attack_name, "Combin", "adaptive_rate")
#     create_dir(save_dir)
#     all_y = []
#     labels = []
#     for class_idx in range(class_num):
#         y_list = dic_1[class_idx]
#         all_y.append(y_list)
#         labels.append(f"Class_{class_idx}")
#     title = f"{dataset_name}_{model_name}_{attack_name}_Combin_adaptive_rate"
#     save_file_name = title+".png"
#     save_path = os.path.join(save_dir, save_file_name)
#     xlabel = "Category"
#     ylabel = "Precision"
#     draw_box(all_y, labels, title, xlabel, ylabel, save_path)
#     print(f"mutated_model_num:{cur_mutated_model_num}")
#     print("get_classes_precision_of_Combination_mutator_by_adpative_mutation_rate() success")



if __name__ == "__main__":
    # 配置数据
    dataset_name = config.dataset_name
    model_name = config.model_name
    attack_name = config.attack_name
    mutation_name_list = config.mutation_name_list 
    mutation_rate_list = config.mutation_rate_list
    exp_root_dir = config.exp_root_dir
    class_num = config.class_num
    mutated_model_num = config.mutation_model_num
    mutation_operator_num = len(mutation_name_list)
    # 第一步 检测target class
    targetClassProcessor = TargetClassProcessor(
        dataset_name, 
        model_name,
        attack_name,
        mutation_name_list,
        mutation_rate_list,
        exp_root_dir,
        class_num,
        mutated_model_num,
        mutation_operator_num)
    # data_dic_0 = targetClassProcessor.get_dict_of_each_mutation_rate_each_classes_precision_list_by_hybrid_mutator()
    # data_dic_1 = targetClassProcessor.get_target_class_with_dif_mutation_rate(data_dic_0)
    # data_dic_2 = targetClassProcessor.get_pValue_Ciff_with_dif_mutation_rate(data_dic_0,data_dic_1)
    # data_dic_3 = targetClassProcessor.get_AdaptiveRate_and_TargetClass(data_dic_2)
    data_dic_3, data_dic_0 = targetClassProcessor.get_adaptive_rate_of_Hybrid_mutator()
    print(data_dic_3)

    # get_classes_precision_of_Hybrid_mutator_by_adpative_mutation_rate()

    # get_classes_precision_of_Combination_mutator_by_adpative_mutation_rate()

    # dic_ans, _ = get_target_class_by_combin_mutator_with_adaptive_rate()
    # print(dic_ans)

    # for mutation_name in mutation_name_list:
    #     dic = get_target_class_by_mutation_name(mutation_name)
    #     print(mutation_name)
    #     for mutaion_rate in mutation_rate_list:
    #         print(mutaion_rate)
    #         print(dic[mutaion_rate])
    # dic = get_target_class_by_hybrid_mutator()
    # print(dic)
    pass

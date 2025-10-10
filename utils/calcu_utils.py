import math
from scipy.stats import wilcoxon,mannwhitneyu,ks_2samp
from cliffs_delta import cliffs_delta

def entropy(data):
    """
    计算信息熵
    :param data: 数据集
    :return: 信息熵
    """
    length = len(data)
    counter = {}
    for item in data:
        counter[item] = counter.get(item, 0) + 1
    ent = 0.0
    for _, cnt in counter.items():
        p = float(cnt) / length
        ent -= p * math.log2(p)
    return ent

def compare_WTL(our_list, baseline_list,expect:str, method:str):
    ans = ""
    # 计算W/T/L
    # Wilcoxon:https://blog.csdn.net/TUTO_TUTO/article/details/138289291
    # Wilcoxon：主要来判断两组数据是否有显著性差异。
    if method == "wilcoxon": # 配对
        statistic, p_value = wilcoxon(our_list, baseline_list) # statistic:检验统计量
    elif method == "mannwhitneyu": # 不配对
        statistic, p_value = mannwhitneyu(our_list, baseline_list) # statistic:检验统计量
    elif method == "ks_2samp":
        statistic, p_value = ks_2samp(our_list, baseline_list) # statistic:检验统计量
    # 如果p_value < 0.05则说明分布有显著差异
    # cliffs_delta：比较大小
    # 如果参数1较小的话，则d趋近-1,0.147(negligible)
    d,res = cliffs_delta(our_list, baseline_list)
    if p_value >= 0.05:
        # 值分布没差别
        ans = "Tie"
        return ans
    else:
        # 值分布有差别
        if expect == "small":
            # 指标越小越好，d越接近-1越好
            if d < 0 and res != "negligible":
                ans = "Win"
            elif d > 0 and res != "negligible":
                ans = "Lose"
            else:
                ans = "Tie"
        else:
            # 指标越大越好，d越接近1越好
            if d > 0 and res != "negligible":
                ans = "Win"
            elif d < 0 and res != "negligible":
                ans = "Lose"
            else:
                ans = "Tie"
    return ans

def compare_avg(our_list, baseline_list):
    our_avg = round(sum(our_list)/len(our_list),3)
    baseline_avg = round(sum(baseline_list)/len(baseline_list),3)
    '''
    if expect == "small":
        if our_avg < baseline_avg:  # 满足期盼
            res = "Win"
        else:
            res = "Lose"
    else:
        if our_avg > baseline_avg:  # 满足期盼
            res = "Win"
        else:
            res = "Lose"
    '''
    return our_avg, baseline_avg

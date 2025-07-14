'''
小的python api测试
'''
import time
import queue
import os
import numpy as np
import torch
from cliffs_delta import cliffs_delta
from scipy.stats import wilcoxon,ks_2samp,mannwhitneyu
from sklearn.metrics import classification_report,precision_score,recall_score,f1_score

def test1():
    print(time.strftime("%Y-%m-%d_%H:%M:%S"))
    q = queue.PriorityQueue()
    q.put((0.0,"szt"))
    q.put((0.0,"fzz"))
def test2():
    os.makedirs("/data/mml2/",exist_ok=True)
def test3():
    d = [1,2,3]
    e = d -1 
    print(e)
def test4():
    a = [4,3,2]
    a.sort() # replace
    print(a)
def test5():
    gt_label = [1,1,1,2,2,2,3,3,3,4,4,4,5,5,5]
    pred_label = [1,2,1,2,3,2,3,4,3,4,3,4,5,4,5]
    ans = classification_report(gt_label,pred_label,output_dict=True)
    print(ans)

def test6():
    timestamp = time.time()
    date_time = time.localtime(timestamp)
    formatted_time = time.strftime('%Y-%m-%d_%H:%M:%S', date_time)
    print(formatted_time)

def test7():
    res = ",".join(["a","b"])
    print(res)

def test8():
    res = [[True]*3]*2
    print(res)

def test9():
    data = [0,1,2]
    data.insert(0,4)
    print(data)
    

def test10():
    data = np.array([4,1,2])
    ranked_idx_array =  data.argsort()
    print("")

def test11():
    data = [9,8]
    a = data[:0]
    data.insert(-4,1)
    print(data)

def test12():
    data = np.array([3,5,1,0,9,8])
    rank = data.argsort()
    print(rank)

def test13():
    # import matplotlib as mpl
    # print(mpl.get_cachedir())
    import matplotlib
    from matplotlib import font_manager
    font_list=sorted([f.name for f in font_manager.fontManager.ttflist])
    for i in font_list:
        print(i)

def test14():
    data = np.array([3,5,1,0,9,8])
    print(3.0 in data)

def test15():
    data = [9]*4
    print(data)

def test17():
    dic = {9:"a",8:"b",0:"c"}
    print(dic.keys())

def test18():
    a= (10.0-2)/10.0
    b = (10-2)/10
    print(a,b)

def test19():
    a = np.array([3,1,5])
    a.argsort()
    print(a)
    np.argsort(a)
    print(a)
    np.sort(a) # no replace
    # a.sort() # replace
    print(a)
    a.tolist()
    print(type(a))

def test20():
    a = np.array([1,2,3])
    print(len(a))
    for i,d in enumerate(a):
        print(i,d)
def test21():
    a = np.array([1,0,1,0])
    x = np.nonzero(a==1)
    print(x)

def test22():
    a = [1,2,3]
    a.insert(-1,5)
    a.insert(0,9)
    print(a)


def test23():
    list_1 = [1,   554,   195,   595,   719,   429,   711,   639,  1038,
         155,  1582,  1665,  2454,  1467,  1870,  2535,  2048,  1530,
        2115,  1915,  3080,  3756,  3164,  2933,  3155,  3064,  3185,
        3602,  3719,  3655,  4592,  4486,  4116,  4510,  4479,  4290,
        4413,  4602,  4313,  3998,  4825,  5865,  5752,  5833,  5379,
        4861,  5199,  4658,  5019,  4676,  6681,  6579,  6078,  7128,
        6573,  7012,  6365,  6554,  6561,  6936,  8300,  7632,  8172,
        7872,  7881,  7612,  8056,  7659,  7423,  8525,  9390,  8588,
        9247,  9187,  8713,  8543,  9777,  9256,  9087,  9204, 10932,
       10342,  9853, 10750, 10822, 11092, 10768, 10052, 10296, 11008,
       12173, 11274, 11245, 12250, 11496, 12245, 12375, 11467, 12011,
       11206, 13623, 13460, 13278, 13310, 12859, 13490, 12718, 12617,
       13616, 13511, 13893, 14070, 14556, 14095, 14519, 14237, 14573,
       14028, 14189, 13954, 15121, 15257, 16075, 15704, 15212, 15231,
       15862, 15146, 15983, 15961, 16812, 16788, 16573, 17587, 17157,
       17155, 17050, 16450, 17620, 16839, 18307, 18669, 18214, 17915,
       18574, 18189, 18721, 18589, 18383, 18636, 19534, 19351, 19955,
       19148, 19066, 19664, 19637, 19647, 18938, 19569, 21014, 20889,
       20873, 20757, 20559, 20268, 21523, 21393, 21024, 21449, 22635,
       21609, 21576, 22233, 22743, 21998, 22723, 22698, 21640, 22266,
       23712, 22853, 23217, 23509, 23855, 23611, 23764, 23684, 22900,
       23962, 25181, 24623, 24950, 24974, 25177, 25041, 25038, 24394,
       24720, 24456, 25678, 25850, 25953, 25680, 26240, 25791, 26186,
       25723, 26353, 26406, 27351, 27708, 27666, 27869, 26968, 26992,
       26834, 27616, 26909, 27119, 28045, 29029, 29076, 29161, 28907,
       29234, 28980, 29134, 28553, 28248, 29699, 29673, 30241, 29437,
       29689, 30333, 30502, 30070, 30433, 30583, 31849, 31708, 30878,
       31435, 31334, 31684, 31860, 30976, 30797, 31417, 32977, 32742,
       31925, 32024, 31977, 33107, 32828, 33106, 32729, 32641, 33647,
       33513, 33792, 33644, 33480, 33719, 33789, 33801, 33536, 34406,
       35159, 35256, 34726, 34686, 35193, 35061, 34961, 35473, 34675,
       34844, 36647, 36466, 36422, 36379, 35963, 36427, 35869, 36549,
       36584, 36392, 37999, 37962, 38240, 37361, 37276, 38208, 38393,
       38035, 37281, 37571]
    list_2 = [ 1, 554, 195, 595, 719, 429, 711, 639, 1038, 155, 1582, 1665, 2454, 1467, 1870, 2535, 2048, 1530, 2115, 1915, 3080, 3756, 3164, 2933, 3155, 3064, 3185, 3602, 3719, 3655, 4592, 4486, 4116, 4510, 4479, 4290, 4413, 4602, 4313, 3998, 4825, 5865, 5752, 5833, 5379, 4861, 5199, 4658, 5019, 4676, 6681, 6579, 6078, 7128, 6573, 7012, 6365, 6554, 6561, 6936, 8300, 7632, 8172, 7872, 7881, 7612, 8056, 7659, 7423, 8525, 9390, 8588, 9247, 9187, 8713, 8543, 9777, 9256, 9087, 9204, 10932, 10342, 9853, 10750, 10822, 11092, 10768, 10052, 10296, 11008, 12173, 11274, 11245, 12250, 11496, 12245, 12375, 11467, 12011, 11206, 13623, 13460, 13278, 13310, 12859, 13490, 12718, 12617, 13616, 13511, 13893, 14070, 14556, 14095, 14519, 14237, 14573, 14028, 14189, 13954, 15121, 15257, 16075, 15704, 15212, 15231, 15862, 15146, 15983, 15961, 16812, 16788, 16573, 17587, 17157, 17155, 17050, 16450, 17620, 16839, 18307, 18669, 18214, 17915, 18574, 18189, 18721, 18589, 18383, 18636, 19534, 19351, 19955, 19148, 19066, 19664, 19637, 19647, 18938, 19569, 21014, 20889, 20873, 20757, 20559, 20268, 21523, 21393, 21024, 21449, 22635, 21609, 21576, 22233, 22743, 21998, 22723, 22698, 21640, 22266, 23712, 22853, 23217, 23509, 23855, 23611, 23764, 23684, 22900, 23962, 25181, 24623, 24950, 24974, 25177, 25041, 25038, 24394, 24720, 24456, 25678, 25850, 25953, 25680, 26240, 25791, 26186, 25723, 26353, 26406, 27351, 27708, 27666, 27869, 26968, 26992, 26834, 27616, 26909, 27119, 28045, 29029, 29076, 29161, 28907, 29234, 28980, 29134, 28553, 28248, 29699, 29673, 30241, 29437, 29689, 30333, 30502, 30070, 30433, 30583, 31849, 31708, 30878, 31435, 31334, 31684, 31860, 30976, 30797, 31417, 32977, 32742, 31925, 32024, 31977, 33107, 32828, 33106, 32729, 32641, 33647, 33513, 33792, 33644, 33480, 33719, 33789, 33801, 33536, 34406, 35159, 35256, 34726, 34686, 35193, 35061, 34961, 35473, 34675, 34844, 36647, 36466, 36422, 36379, 35963, 36427, 35869, 36549, 36584, 36392, 37999, 37962, 38240, 37361, 37276, 38208, 38393, 38035, 37281, 37571]

    diff = list(set(list_1) - set(list_2))
    print(diff)

def test24():
    matrix = [[1,2,3],[4,5,6]]
    x = [row[1] for row in matrix]

def test25():
    data = np.array([0,1,1,0])
    x = np.nonzero(data==1)
    print(x)

def test26():
    a = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
    b = torch.tensor([[1,3,2],[7,4,3],[5,8,9]])
    c = a*b
    print(c)

def test27():
    data_list = [7,3,4]
    data_list.insert(-1,99)
    print(data_list)

def test28():
    res = 0 % 5
    print(res)

def test29():
    idx_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    np.random.seed(2)
    choiced_list = np.random.choice(idx_list, replace=False, size=5).tolist()
    print(choiced_list)

def test30():
    a = torch.tensor([1,2,3])
    print(type(a))

def test31():
    class Person:
        def __init__(self, name, age, sex):
            self.name = name
            self.aget = age
            self.sext = sex
    p1 = Person("mml","30","man")
    print(p1)

def test32():
    data_1_list = list(range(1,10))
    data_2_list = list(range(10,20))

    # 参数1远远小于参数2的话 d=-1 res=large
    d, res =cliffs_delta(data_2_list, data_1_list)
    print(d,res)

def test33():
    '''
    # Scene:CIFAR10|ResNet18|IAD
    our_asr_list = [0.017, 0.058, 0.029, 0.046, 0.082, 0.033, 0.082, 0.045, 0.03, 0.102]
    asd_asr_list = [0.046, 0.048, 0.997, 1.0, 1.0, 0.999, 0.034, 0.996, 0.999, 0.05]
    '''
    
    # Scene:CIFAR10|ResNet18|Refool
    our_asr_list = [0.792, 0.389, 0.014, 0.002, 0.03, 0.015, 0.104, 0.005, 0.012, 0.029]
    asd_asr_list = [0.195, 0.302, 0.177, 0.51, 0.149, 0.271, 0.11, 0.688, 0.414, 0.071]
    
    s,p =  ks_2samp(our_asr_list, asd_asr_list)
    statistic, p_value = wilcoxon(our_asr_list, asd_asr_list) # statistic:检验统计量
    # 原假设H0：两个独立样本来自相同的总体（或两个总体的分布相同，没有位置偏移）。
    s_m,m_p =  mannwhitneyu(our_asr_list,asd_asr_list)
    print(f"原始：ks:{p}, wil:{p_value}, m:{m_p}")

    data_list_1 = our_asr_list*100
    data_list_2 = asd_asr_list*100

    s,p =  ks_2samp(data_list_1, data_list_2)
    statistic, p_value = wilcoxon(data_list_1, data_list_2) # statistic:检验统计量
    s_m,m_p =  mannwhitneyu(data_list_1,data_list_2)
    print(f"10*原始：ks:{p}, wil:{p_value}, m:{m_p}")



    '''
    data_list_1 = [1,2,3,4,5,6,7,8,9,10]
    data_list_2 = [1,2,3,4,5,6,7,8,9,10]
    s,p =  ks_2samp(data_list_1, data_list_2)
    statistic, p_value = wilcoxon(data_list_1, data_list_2) # statistic:检验统计量
    print(f"相同：ks:{p}, wil:{p_value}")
    '''

    '''

    data_list_1 = [1,2,3,4,5,6,7,8,9,10]*10
    data_list_2 = [1,2,3,4,5,6,7,8,9,10]*10
    s,p =  ks_2samp(data_list_1, data_list_2)
    statistic, p_value = wilcoxon(data_list_1, data_list_2) # statistic:检验统计量
    print(f"10*相同：ks:{p}, wil:{p_value}")
    '''

    data_list_1 = [1,2,3,4,5,6,7,8,9,10]
    data_list_2 = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.1]

    s,p =  ks_2samp(data_list_1, data_list_2)
    statistic, p_value = wilcoxon(data_list_1, data_list_2) # statistic:检验统计量
    s_m,m_p =  mannwhitneyu(data_list_1,data_list_2)
    print(f"明显差异：ks:{p}, wil:{p_value}, m:{m_p}")



    # 如果p_value < 0.05则说明分布有显著差异
    # cliffs_delta：比较大小
    # 如果参数1较小的话，则d趋近-1,0.147(negligible)
    d,res = cliffs_delta(our_asr_list, asd_asr_list)
    print("")


def test34():
    s = 0
    for _ in range(6):
        b = 9
        a = 7
        s += (b-a)
    print(s)

def test35():
    d = torch.stack([1,2,3])
    print(d)

def test36():
    set_a = {1,2,3}
    b = set(set_a)
    print()

def test37():
    import matplotlib.pyplot as plt
    import numpy as np

    # 示例数据 - 11个类别，每个类别10个堆叠部分
    categories = [f'类别 {i}' for i in range(1, 12)]  # 11个类别
    num_stacks = 10  # 每个柱子的堆叠部分数量

    # 创建随机数据（替换为您的实际数据）
    # 每个类别有10个堆叠部分的数据
    np.random.seed(42)  # 确保可重复性
    stack_data = np.random.randint(5, 25, size=(num_stacks, len(categories)))

    # 创建颜色映射（使用tab10色图，可自定义）
    colors = plt.cm.tab10(np.linspace(0, 1, num_stacks))

    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(15, 8))

    # 初始化底部位置（从0开始）
    bottoms = np.zeros(len(categories))

    # 绘制堆叠柱状图
    for i in range(num_stacks):
        ax.bar(categories, stack_data[i], bottom=bottoms, 
            color=colors[i], label=f'部分 {i+1}')
        bottoms += stack_data[i]  # 更新底部位置

    # 添加总量标签（每个柱子的顶部）
    for i, total in enumerate(bottoms):
        ax.text(i, total + 0.5, f'{int(total)}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 设置图表标题和标签
    ax.set_title('11个类别的数据分布（10部分堆叠柱状图）', fontsize=16, pad=20)
    ax.set_xlabel('类别', fontsize=12)
    ax.set_ylabel('总数量', fontsize=12)

    # 添加图例（放在图表外部右侧）
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title='堆叠部分')

    # 调整布局
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # 为图例留出空间
    plt.grid(axis='y', alpha=0.3)

    # 显示图表
    plt.show()
    plt.savefig("imgs/temp.png")

if __name__ == "__main__":
    test37()



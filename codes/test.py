'''
小的python api测试
'''
import time
import queue
import os
import numpy as np

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


if __name__ == "__main__":
    test15()




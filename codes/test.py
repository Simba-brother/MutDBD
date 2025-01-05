import time
import queue
import os
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

if __name__ == "__main__":
    test5()




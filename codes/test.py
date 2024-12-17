import time
import queue
import os

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
if __name__ == "__main__":
    test4()




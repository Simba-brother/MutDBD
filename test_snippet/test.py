
import os
import json
from collections import defaultdict
def test1():
    data = [0,1,2]
    data.insert(-1,99)
    print(data)

def test2():
    # 三层嵌套字典
    tree = lambda: defaultdict(tree)
    tree_data = tree()
    tree_data['a']['b']['c'] = 1
    with open("tree.json", 'w') as f:
        json.dump(tree_data, f)
def test3():
    p = os.path.join("/home/mml","w","a/b")
    print(p)
if __name__ == "__main__":
    test3()

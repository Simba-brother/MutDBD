from collections import defaultdict
from queue import PriorityQueue

def nested_defaultdict(depth, default_factory=int):
    if depth == 1:
        return defaultdict(default_factory)
    else:
        return defaultdict(lambda: nested_defaultdict(depth - 1, default_factory))
    
def priorityQueue_to_list(pq:PriorityQueue) -> list:
    _list = []
    while not pq.empty():
        _list.append(pq.get())
    return _list

from collections import defaultdict
def nested_defaultdict(depth, default_factory=int):
    if depth == 1:
        return defaultdict(default_factory)
    else:
        return defaultdict(lambda: nested_defaultdict(depth - 1, default_factory))
import torch
print("test3 global code")
class Person(object):
    def __init__(self):
        self.name = "mml"
def f1():
    print("test3 f1()被运行")

def save_Person():
    p  = Person()
    torch.save(p, "test3_p.pth")

def load_Person():
    p =  torch.load("test3_p.pth")
    return p
if __name__ == "__main__":
    # save_Person()
    load_Person()
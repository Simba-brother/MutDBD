import torch
print("test4 global code")
class Person(object):
    def __init__(self):
        self.id = "1"
def f1():
    print("test4 f1()被运行")

def save_Person():
    p  = Person()
    torch.save(p, "test4_p.pth")

def load_Person():
    p =  torch.load("test4_p.pth")
    return p
if __name__ == "__main__":
    save_Person()
    # load_Person()
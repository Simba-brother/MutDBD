import torch
import numpy as np
def test1():
    tensor_1 = torch.ones([2,3])
    print(tensor_1.shape)
    tensor_1_flatten = tensor_1.flatten()
    tensor_1_flatten[0] = 2
    print(tensor_1)
    print("test1")
def test2():
    a = "mml"
    print(len(a))
def test3():
    data = torch.tensor([0,1,0,1])
    a = np.nonzero(data == 1)
    print("jfla")

if __name__ == "__main__":
    test3()
import torch
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


if __name__ == "__main__":
    test2()
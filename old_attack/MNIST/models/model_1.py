import torch
import torch.nn as nn
class CNN_Model_1(nn.Module):
    def __init__(self, class_num):
        super(CNN_Model_1, self).__init__()
        # convolution 1
        self.c1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5,5), stride=1, padding=0)
        self.relu1 = nn.ReLU()
        # maxpool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2))
        # dropout 1
        self.dropout1 = nn.Dropout(0.25)
        # convolution 2
        self.c2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=1, padding=0)
        self.relu2 = nn.ReLU()
        # maxpool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2))
        # dropout 2
        self.dropout2 = nn.Dropout(0.25)
        # linear 1
        self.fc1 = nn.Linear(32*5*5, 256)
        # dropout 3
        self.dropout3 = nn.Dropout(0.25)
        # linear 2
        self.fc2 = nn.Linear(256, class_num)
        
    def forward(self, x):
        
        out = self.c1(x) # [BATCH_SIZE, 16, 24, 24]
        out = self.relu1(out) 
        out = self.maxpool1(out) # [BATCH_SIZE, 16, 12, 12]
        out = self.dropout1(out) 
        
        out = self.c2(out) # [BATCH_SIZE, 32, 10, 10]
        out = self.relu2(out) 
        out = self.maxpool2(out) # [BATCH_SIZE, 32, 5, 5]
        out = self.dropout2(out) 
        
        out = out.view(out.size(0), -1) # [BATCH_SIZE, 32*5*5=800]
        out = self.fc1(out) # [BATCH_SIZE, 256]
        out = self.dropout3(out)
        out = self.fc2(out) # [BATCH_SIZE, 10]
        
        return out
    
if __name__ == "__main__":
    model = CNN_Model_1(class_num=10)
    x = torch.randn(16, 1, 28, 28)
    x = model(x)
    print(x.size())
    print(x)
'''
重要
专门用于模型的评估
'''
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from utils.common_utils import set_random_seed, read_yaml
config = read_yaml("config.yaml")

set_random_seed(config["global_random_seed"])

class EvalModel(object):
    def __init__(self, model, dataset, device, batch_size=512, num_workers=8):
        # 3个属性：模型，数据集和设备
        self.model = model
        self.device = device
        self.dataset_loader = self.get_dataset_loader(dataset,batch_size,num_workers)

    def get_dataset_loader(self, dataset,batch_size,num_workers):
        dataset_loader = DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True
        )
        return dataset_loader

    def eval_acc(self):
        '''
        评估模型的accuracy
        '''
        self.model.to(self.device)
          # put network in train mode for Dropout and Batch Normalization
        self.model.eval()
        acc = torch.tensor(0., device=self.device) 
        total_num = 0
        correct_num = 0 
        with torch.no_grad():
            for batch_id, batch in enumerate(self.dataset_loader):
                X = batch[0]
                Y = batch[1]
                X = X.to(self.device)
                Y = Y.to(self.device)
                preds = self.model(X)
                # _, predicted = preds.max(1)
                # total_num += Y.size(0)
                # correct_num += predicted.eq(Y).sum().item()
                correct_num += (torch.argmax(preds, dim=1) == Y).sum()
                total_num += X.shape[0]
        acc = correct_num/total_num
        acc = round(acc.item(),3)
        return acc

    def eval_TrueOrFalse(self):
        '''
        评估TrueorFalse结果
        '''
        self.model.to(self.device)
        self.model.eval()
        trueOrFalse_list = []
        with torch.no_grad():
            for batch_id, batch in enumerate(self.dataset_loader):
                X = batch[0]
                Y = batch[1]
                X = X.to(self.device)
                Y = Y.to(self.device)
                preds = self.model(X)
                trueOrFalse_list.extend((torch.argmax(preds, dim=1) == Y).tolist()) 
        return trueOrFalse_list
    
    def eval_classification_report(self):
        '''
        获得classification_report
        '''
        self.model.to(self.device)
        self.model.eval()  # put network in train mode for Dropout and Batch Normalization
        pred_labels = []
        true_labels = []
        with torch.no_grad():
            for batch_id, batch in enumerate(self.dataset_loader):
                X = batch[0]
                Y = batch[1]
                X = X.to(self.device)
                Y = Y.to(self.device)
                outputs = self.model(X)
                pred_labels.extend(torch.argmax(outputs,dim=1).tolist()) 
                true_labels.extend(Y.tolist()) 
        report = classification_report(true_labels, pred_labels, output_dict=True)
        return report
    
    def get_pred_labels(self):
        '''
        得到预测的标签
        '''
        self.model.to(self.device)
        self.model.eval()
        pred_labels = []
        # true_labels = []
        with torch.no_grad():
            for batch_id, batch in enumerate(self.dataset_loader):
                X = batch[0]
                # Y = batch[1]
                X = X.to(self.device)
                # Y = Y.to(self.device)
                preds = self.model(X)
                pred_labels.extend(torch.argmax(preds,dim=1).tolist()) 
                # true_labels.extend(Y.tolist()) 
        return pred_labels
    
    def get_outputs(self):
        '''
        得到输出值
        '''
        # 评估开始时间
        start = time.time()
        self.model.to(self.device)
        self.model.eval()
        outputs = []
        with torch.no_grad():
            for batch_id, batch in enumerate(self.dataset_loader):
                X = batch[0]
                Y = batch[1]
                X = X.to(self.device)
                Y = Y.to(self.device)
                preds = self.model(X)
                outputs.extend(preds.tolist()) 
        end = time.time()
        print("cost time:", end-start)
        return outputs
    
    def get_prob_outputs(self):
        '''
        得到概率输出值
        '''
        self.model.to(self.device)
        self.model.eval()  # put network in eval mode for Dropout and Batch Normalization
        outputs = []
        with torch.no_grad():
            for batch_id, batch in enumerate(self.dataset_loader):
                X = batch[0]
                Y = batch[1]
                X = X.to(self.device)
                Y = Y.to(self.device)
                preds = self.model(X)
                probability = F.softmax(preds, dim=1)
                outputs.extend(probability.tolist())
        return outputs
    
    def get_CEloss(self):
        '''
        得到交叉熵损失值
        '''
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.model.to(self.device)
        self.model.eval()  # put network in eval mode for Dropout and Batch Normalization
        # 存储该模型在训练集样本上的交叉熵损失
        CE_loss = []
        with torch.no_grad():
            for batch_id, batch in enumerate(self.dataset_loader):
                X = batch[0]
                Y = batch[1]
                X = X.to(self.device)
                Y = Y.to(self.device)
                # 模型最后一层输出(可以为非概率分布形式)
                outputs = self.model(X)
                loss_ce = criterion(outputs,Y) 
                CE_loss.extend(loss_ce.tolist())
        return CE_loss

    
    def get_confidence_list(self):
        '''
        得到top1 confidence list
        '''
        self.model.to(self.device)
        self.model.eval()  # put network in train mode for Dropout and Batch Normalization
        confidence_list = []
        with torch.no_grad():
            for batch_id, batch in enumerate(self.dataset_loader):
                X = batch[0]
                Y = batch[1]
                X = X.to(self.device)
                Y = Y.to(self.device)
                preds = self.model(X)
                probability = F.softmax(preds, dim=1)
                vaules, indices = torch.max(probability,dim=1)
                confidence_list.extend(vaules.tolist())
        return confidence_list

def eval_asr_acc(model,poisoned_set,clean_set,device):
    e = EvalModel(model,poisoned_set,device)
    asr = e.eval_acc()
    e = EvalModel(model,clean_set,device)
    acc = e.eval_acc()
    return asr,acc
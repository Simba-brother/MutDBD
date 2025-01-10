'''
专门用于模型的评估
'''
import time
import torch
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from codes.config import random_seed


torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

def _seed_worker(worker_id):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

class EvalModel(object):
    def __init__(self, model, testset, device):
        # 3个属性：模型，数据集和设备
        self.model = model 
        self.testset = testset
        self.device = device

    def eval_acc(self,batch_size=128):
        '''
        评估模型的accuracy
        '''
        testset_loader = DataLoader(
            self.testset,
            batch_size = batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            pin_memory=False,
            worker_init_fn=_seed_worker
        )
        self.model.to(self.device)
          # put network in train mode for Dropout and Batch Normalization
        self.model.eval()
        acc = torch.tensor(0., device=self.device) 
        total_num = len(self.testset)
        correct_num = 0 
        with torch.no_grad():
            for batch_id, batch in enumerate(testset_loader):
                X = batch[0]
                Y = batch[1]
                X = X.to(self.device)
                Y = Y.to(self.device)
                preds = self.model(X)
                correct_num += (torch.argmax(preds, dim=1) == Y).sum()
        acc = correct_num/total_num
        acc = round(acc.item(),3)
        return acc

    def eval_TrueOrFalse(self,batch_size=128):
        '''
        评估TrueorFalse结果
        '''
        testset_loader = DataLoader(
            self.testset,
            batch_size = batch_size,
            shuffle=False,
            # num_workers=self.current_schedule['num_workers'],
            drop_last=False,
            pin_memory=False,
            worker_init_fn=_seed_worker
        )
        
        self.model.to(self.device)
        self.model.eval()
        trueOrFalse_list = []
        with torch.no_grad():
            for batch_id, batch in enumerate(testset_loader):
                X = batch[0]
                Y = batch[1]
                X = X.to(self.device)
                Y = Y.to(self.device)
                preds = self.model(X)
                trueOrFalse_list.extend((torch.argmax(preds, dim=1) == Y).tolist()) 
        return trueOrFalse_list
    
    def eval_classification_report(self,batch_size =128):
        '''
        获得classification_report
        '''
        testset_loader = DataLoader(
            self.testset,
            batch_size = batch_size,
            shuffle=False,
            num_workers= 1,
            drop_last=False,
            pin_memory=False
            # worker_init_fn=_seed_worker
        )
        self.model.to(self.device)
        self.model.eval()  # put network in train mode for Dropout and Batch Normalization
        pred_labels = []
        true_labels = []
        with torch.no_grad():
            for batch_id, batch in enumerate(testset_loader):
                X = batch[0]
                Y = batch[1]
                X = X.to(self.device)
                Y = Y.to(self.device)
                outputs = self.model(X)
                pred_labels.extend(torch.argmax(outputs,dim=1).tolist()) 
                true_labels.extend(Y.tolist()) 
        report = classification_report(true_labels, pred_labels, output_dict=True)
        return report
    
    def get_pred_labels(self,batch_size =128):
        '''
        得到预测的标签
        '''
        testset_loader = DataLoader(
            self.testset,
            batch_size = batch_size,
            shuffle=False,
            # num_workers=self.current_schedule['num_workers'],
            drop_last=False,
            pin_memory=False,
            worker_init_fn=_seed_worker
        )
        self.model.to(self.device)
        self.model.eval()
        pred_labels = []
        # true_labels = []
        with torch.no_grad():
            for batch_id, batch in enumerate(testset_loader):
                X = batch[0]
                # Y = batch[1]
                X = X.to(self.device)
                # Y = Y.to(self.device)
                preds = self.model(X)
                pred_labels.extend(torch.argmax(preds,dim=1).tolist()) 
                # true_labels.extend(Y.tolist()) 
        return pred_labels
    
    def get_outputs(self,batch_size =128):
        '''
        得到输出值
        '''
        testset_loader = DataLoader(
            self.testset,
            batch_size = batch_size,
            shuffle=False,
            # num_workers=self.current_schedule['num_workers'],
            drop_last=False,
            pin_memory=False,
            worker_init_fn=_seed_worker
        )
        # 评估开始时间
        start = time.time()
        self.model.to(self.device)
        self.model.eval()
        outputs = []
        with torch.no_grad():
            for batch_id, batch in enumerate(testset_loader):
                X = batch[0]
                Y = batch[1]
                X = X.to(self.device)
                Y = Y.to(self.device)
                preds = self.model(X)
                outputs.extend(preds.tolist()) 
        end = time.time()
        print("cost time:", end-start)
        return outputs
    
    def get_prob_outputs(self,batch_size =128):
        '''
        得到概率输出值
        '''
        testset_loader = DataLoader(
            self.testset,
            batch_size = batch_size,
            shuffle=False,
            # num_workers=self.current_schedule['num_workers'],
            drop_last=False,
            pin_memory=False,
            worker_init_fn=_seed_worker
        )
        # 评估开始时间
        self.model.to(self.device)
        self.model.eval()  # put network in train mode for Dropout and Batch Normalization
        outputs = []
        with torch.no_grad():
            for batch_id, batch in enumerate(testset_loader):
                X = batch[0]
                Y = batch[1]
                X = X.to(self.device)
                Y = Y.to(self.device)
                preds = self.model(X)
                probability = F.softmax(preds, dim=1)
                outputs.extend(probability.tolist())
        return outputs
    
    def get_CEloss(self,batch_size =128):
        '''
        得到交叉熵损失值
        '''
        testset_loader = DataLoader(
            self.testset,
            batch_size = batch_size,
            shuffle=False,
            # num_workers=self.current_schedule['num_workers'],
            drop_last=False,
            pin_memory=False,
            worker_init_fn=_seed_worker
        )
        # 评估开始时间
        self.model.to(self.device)
        self.model.eval()  # put network in train mode for Dropout and Batch Normalization
        CE_loss = []
        with torch.no_grad():
            for batch_id, batch in enumerate(testset_loader):
                X = batch[0]
                Y = batch[1]
                X = X.to(self.device)
                Y = Y.to(self.device)
                # 模型最后一层输出(可以为非概率分布形式)
                outputs = self.model(X)
                loss_ce_list = []
                for i in range(outputs.shape[0]):
                    output = outputs[i]
                    y = Y[i]
                    criterion = torch.nn.CrossEntropyLoss()
                    loss_ce = criterion(output,y)
                    loss_ce_list.append(loss_ce.item())
                CE_loss.extend(loss_ce_list)
        return CE_loss

    
    def get_confidence_list(self,batch_size =128):
        '''
        得到top1 confidence list
        '''
        testset_loader = DataLoader(
            self.testset,
            batch_size = batch_size,
            shuffle=False,
            # num_workers=self.current_schedule['num_workers'],
            drop_last=False,
            pin_memory=False,
            worker_init_fn=_seed_worker
        )
        self.model.to(self.device)
        self.model.eval()  # put network in train mode for Dropout and Batch Normalization
        confidence_list = []
        with torch.no_grad():
            for batch_id, batch in enumerate(testset_loader):
                X = batch[0]
                Y = batch[1]
                X = X.to(self.device)
                Y = Y.to(self.device)
                preds = self.model(X)
                probability = F.softmax(preds, dim=1)
                vaules, indices = torch.max(probability,dim=1)
                confidence_list.extend(vaules.tolist())
        return confidence_list


    
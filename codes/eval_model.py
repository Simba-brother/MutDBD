import sys
sys.path.append("./")
from collections import defaultdict
import time
import torch
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

global_seed = 666
torch.manual_seed(global_seed)
np.random.seed(global_seed)
random.seed(global_seed)

def _seed_worker():
    global_seed = 666
    torch.manual_seed(global_seed)
    np.random.seed(global_seed)
    random.seed(global_seed)

class EvalModel(object):
    def __init__(self, model, testset, device):
        self.model = model
        self.testset = testset
        self.device = device

    def _eval_acc(self):
        batch_size =128
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
        self.model.eval()  # put network in train mode for Dropout and Batch Normalization
        acc = torch.tensor(0., device=self.device) # 攻击成功率
        total_num = len(self.testset)
        correct_num = 0 # 攻击成功数量
        with torch.no_grad():
            for X, Y in testset_loader:
                X = X.to(self.device)
                Y = Y.to(self.device)
                preds = self.model(X)
                correct_num += (torch.argmax(preds, dim=1) == Y).sum()
        acc = correct_num/total_num
        acc = round(acc.item(),3)
        end = time.time()
        print(f"Total time consumption:{end-start}s")
        return acc

    def _get_TrueOrFalse(self):
        batch_size =128
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
        self.model.eval()  # put network in train mode for Dropout and Batch Normalization
        trueOrFalse_list = []
        with torch.no_grad():
            for X, Y in testset_loader:
                X = X.to(self.device)
                Y = Y.to(self.device)
                preds = self.model(X)
                trueOrFalse_list.extend((torch.argmax(preds, dim=1) == Y).tolist()) 
        end = time.time()
        print(f"Total time consumption:{end-start}s")
        return trueOrFalse_list
    
    def _eval_classes_acc(self):
        batch_size =128
        testset_loader = DataLoader(
            self.testset,
            batch_size = batch_size,
            shuffle=False,
            num_workers= 1, #self.current_schedule['num_workers'],
            drop_last=False,
            pin_memory=False
            # worker_init_fn=_seed_worker
        )
        # 评估开始时间
        start = time.time()
        self.model.to(self.device)
        self.model.eval()  # put network in train mode for Dropout and Batch Normalization
        pred_labels = []
        true_labels = []
        # {class_idx:correct_num}
        # correct_count_dict = defaultdict(int)
        # total_count_dict = defaultdict(int)
        # acc_dict = defaultdict(int)
        with torch.no_grad():
            for X, Y in testset_loader:
                X = X.to(self.device)
                Y = Y.to(self.device)
                outputs = self.model(X)
                # predicted_labels = torch.argmax(outputs,dim=1)
                # for label,predicted_label in zip(Y, predicted_labels):
                #     label = label.item()
                #     predicted_label = predicted_label.item()
                #     total_count_dict[label] += 1
                #     if label == predicted_label:
                #         correct_count_dict[label] += 1
                pred_labels.extend(torch.argmax(outputs,dim=1).tolist()) 
                true_labels.extend(Y.tolist()) 
        end = time.time()
        print("_eval_classes_acc() cost time:", end-start)
        report = classification_report(true_labels, pred_labels, output_dict=True)
        # for class_idx, correct_count in correct_count_dict.items():
        #     total_count = total_count_dict[class_idx]
        #     acc = correct_count/total_count
        #     acc = round(acc,3)
        #     acc_dict[class_idx] = acc
        return report
    
    def _get_pred_labels(self):
        batch_size =128
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
        self.model.eval()  # put network in train mode for Dropout and Batch Normalization
        pred_labels = []
        true_labels = []
        with torch.no_grad():
            for X, Y in testset_loader:
                X = X.to(self.device)
                Y = Y.to(self.device)
                preds = self.model(X)
                pred_labels.extend(torch.argmax(preds,dim=1).tolist()) 
                true_labels.extend(Y.tolist()) 
        end = time.time()
        print("cost time:", end-start)
        return pred_labels
    
    def _get_outputs(self):
        batch_size =128
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
        self.model.eval()  # put network in train mode for Dropout and Batch Normalization
        outputs = []
        with torch.no_grad():
            for X, Y in testset_loader:
                X = X.to(self.device)
                Y = Y.to(self.device)
                preds = self.model(X)
                outputs.extend(preds.tolist()) 
        end = time.time()
        print("cost time:", end-start)
        return outputs
    
    def _get_prob_outputs(self):
        batch_size =128
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
        self.model.eval()  # put network in train mode for Dropout and Batch Normalization
        outputs = []
        with torch.no_grad():
            for X, Y in testset_loader:
                X = X.to(self.device)
                Y = Y.to(self.device)
                preds = self.model(X)
                probability = F.softmax(preds, dim=1)
                outputs.extend(probability.tolist()) 
        end = time.time()
        print("cost time:", end-start)
        return outputs
    
    def _get_confidence_list(self):
        batch_size =128
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
        self.model.eval()  # put network in train mode for Dropout and Batch Normalization
        confidence_list = []
        with torch.no_grad():
            for X, Y in testset_loader:
                X = X.to(self.device)
                Y = Y.to(self.device)
                preds = self.model(X)
                probability = F.softmax(preds, dim=1)
                vaules, indices = torch.max(probability,dim=1)
                confidence_list.extend(vaules.tolist())
        end = time.time()
        print("cost time:", end-start)
        return confidence_list

# from codes.datasets.cifar10.attacks.badnets_resnet18_nopretrain_32_32_3 import PureCleanTrainDataset, PurePoisonedTrainDataset, get_dict_state
# origin_dict_state = get_dict_state()
# backdoor_model = origin_dict_state["backdoor_model"]
# clean_testset = origin_dict_state["clean_testset"]
# poisoned_testset = origin_dict_state["poisoned_testset"]
# pureCleanTrainDataset = origin_dict_state["pureCleanTrainDataset"]
# purePoisonedTrainDataset = origin_dict_state["purePoisonedTrainDataset"]
# poisoned_trainset = origin_dict_state["poisoned_trainset"]
# device = torch.device("cuda:0")
# if __name__ == "__main__":
#     e = EvalModel(backdoor_model, clean_testset, device)
#     report = e._eval_classes_acc()
    
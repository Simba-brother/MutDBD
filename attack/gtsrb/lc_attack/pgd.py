import torch
class PGD():
    def __init__(self, net, loss_fn, steps:int, alpha:float, epsilon:float):
        self.net = net
        self.loss_fn = loss_fn
        self.steps = steps
        self.alpha = alpha
        self.epsilon = epsilon
        self.name = "pgd"

    def perturb(self, X, Y, device=None):
        """ generates adversarial examples to given data points and labels (X,Y) based on PGD approach. """
        '''X:shape:(BCHW),type:Tensor,数值归一化了'''
        original_X = X.clone().detach().to(device)
        # 创建对抗样本的初始副本
        X_adv = X.clone().detach().to(device)
        Y= Y.to(device)
        for step_i in range(self.steps):
            X_adv.requires_grad = True
            outputs = self.net(X_adv)
            _loss = self.loss_fn(outputs, Y)
            # 计算梯度
            self.net.zero_grad()
            _loss.backward()
            # 更新对抗样本
            with torch.no_grad():
                # 使用梯度符号更新
                X_adv = X_adv + self.alpha * X_adv.grad.sign()
                # 确保扰动在 ε 范围内
                delta = torch.clamp(X_adv - original_X, min=-self.epsilon, max=self.epsilon)
                X_adv = torch.clamp(original_X + delta, min=0.0, max=1.0).detach()
        return X_adv
    

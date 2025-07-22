import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, pred_mask, gt_mask):
        # 将预测掩码转换为概率
        pred_prob = torch.sigmoid(pred_mask)
        
        # 计算交叉熵损失
        ce_loss = F.binary_cross_entropy(pred_prob, gt_mask, reduction='none')
        
        # 计算Focal Loss
        p_t = pred_prob * gt_mask + (1 - pred_prob) * (1 - gt_mask)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        # 返回平均损失
            # bce_loss = F.binary_cross_entropy_with_logits(gt_mask, gt_mask, reduction='none')
        
        # # 计算预测概率
        # pred_prob = torch.sigmoid(gt_mask)
        
        # # 计算Focal Loss的加权因子
        # pt = torch.where(gt_mask == 1, pred_prob, 1 - pred_prob)
        # focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # # 计算Focal Loss
        # focal_loss = focal_weight * bce_loss
        return focal_loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 将输入和目标展平
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # 计算交集和并集
        intersection = (inputs * targets).sum()
        total = inputs.sum() + targets.sum()
        
        # 计算 Dice 系数
        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
        
        # 计算 Dice Loss
        loss = 1.0 - dice
        return loss
    
class IoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 将输入和目标展平
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # 计算交集和并集
        intersection = (inputs * targets).sum()
        total = inputs.sum() + targets.sum()
        union = total - intersection
        
        # 计算 IoU
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        # 计算 IoU Loss
        loss = 1.0 - iou
        return loss
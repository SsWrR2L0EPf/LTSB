import torch
import torch.nn as nn
import torch.nn.functional as F


class LTSBLoss(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.num_classes = num_classes

    def cal_weight_for_classes(self, cls_num_list):
        cls_num_list = torch.Tensor(cls_num_list)
        weight = torch.log(cls_num_list / cls_num_list.sum() + 1e-9)[None, :]
        self.weight = weight.cuda()

    def cls_loss(self, logits, logits2, sim, targets, targets_con):
        cross_entropy_loss = F.cross_entropy(logits, targets)

        con_loss = F.cross_entropy(sim, targets_con)

        loss = cross_entropy_loss*4 + con_loss*1
        return loss

    def forward(self, logits, logits2, sim, targets, targets_con):
        loss = self.cls_loss(logits, logits2, sim, targets, targets_con)
        return loss
